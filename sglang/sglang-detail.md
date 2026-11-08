# SGLang Parameters for Large-MoE RL Rollout

> 逐条拆解 256 H100 上 TB 级大规模 MoE 模型（适用 DeepSeek / Qwen 架构，稀疏 attention + 内置 MTP）的 RL 训练脚本里每个 sglang 参数：语义、调参方向、坑点、代码引用。讲解对象为本机 `/Users/roy/Desktop/sglang`（main 分支）。体系性讲解见 `sglang-overview.md`；`--sglang-*` 前缀透传、`tp_size = num_gpus_per_engine`、mem-fraction 基础见 `2-sglang.md`。DeepGEMM、DeepEP、FlashMLA（NSA/DSA）均为 DeepSeek 的开源工作，首次出现处给出代码/论文链接。

## Contents

- [Basic: How the GPU Actually Runs a Forward Pass](#basic-how-the-gpu-actually-runs-a-forward-pass)
  - [Networking: NVLink, InfiniBand, RDMA](#networking-nvlink-infiniband-rdma)
  - [Compute and Compilation: Kernel, Eager, CUDA Graph, JIT](#compute-and-compilation-kernel-eager-cuda-graph-jit)
  - [THD Data Layout and Multi-Request Batching](#thd-data-layout-and-multi-request-batching)
- [Request Scheduling: Two Layers of Load Balancing](#request-scheduling-two-layers-of-load-balancing)
- [Config at a Glance](#config-at-a-glance)
- [Parallelism and MoE Communication](#parallelism-and-moe-communication)
- [PD Disaggregation](#pd-disaggregation)
- [CUDA Graph and Scheduling](#cuda-graph-and-scheduling)
- [KV Cache](#kv-cache)
- [Speculative Decoding](#speculative-decoding)

---

## Basic: How the GPU Actually Runs a Forward Pass

调参前先补三块底层知识：数据怎么跨卡传（网络）、kernel 怎么执行和编译（计算）、多请求怎么打包进一次 forward（数据布局）。后面所有参数都往这三节回溯。

### Networking: NVLink, InfiniBand, RDMA

PD 分离要跨机器传 KV cache，DeepEP 要跨机器做 all-to-all。跨机器这件事，普通以太网做不了——延迟几十微秒加操作系统开销，KV 传输会拖垮 decode。这一节讲清楚数据在集群里怎么走。

#### NVLink Within Node, IB Across Nodes

一台机器（节点）里 8 张 GPU 通过 NVLink 直连，H100 NVLink 总带宽 900 GB/s，延迟纳秒级。跨机器没有 NVLink，只有网络。

InfiniBand（IB）是数据中心专用高速网络，不是普通以太网。每端口 200 Gb/s（HDR）或 400 Gb/s（NDR），H100 集群通常每节点 8 张 IB 网卡，聚合 1.6 Tb/s；端到端延迟 1–3 微秒，而 TCP 是 10–100 微秒。IB 网卡驱动名是 `mlx5_*`（第五代驱动名，与带宽无关）。

#### RDMA Bypasses the CPU

RDMA（Remote Direct Memory Access）是 IB 的核心能力：数据从 GPU A 的网卡直接进 GPU B 的显存，完全绕过两端的 CPU 和操作系统，延迟微秒级，CPU 占用接近 0。对比普通网络——GPU A → CPU A 内存 → OS → 网卡 → 网络 → 网卡 → OS → CPU B → GPU B，每一跳都有 CPU 参与。PD 分离里一条请求的 KV cache 约 159 MB，RDMA 走 200 GB/s 只要 0.8 ms，换 TCP 要几十毫秒，不可接受。

RDMA 不等于 NCCL。NCCL 是通信库，按拓扑自动选底层通道；RDMA 是底层通道之一：

```
NCCL
├─ 节点内 → NVLink（首选）/ PCIe
└─ 节点间 → IB/RoCE（RDMA）/ TCP（兜底）
```

DeepEP 和 Mooncake 不走 NCCL，直接调 RDMA 原语（NVSHMEM / 自定义 kernel）。原因是 NCCL 的 all-to-all 面向对称固定大小的消息，而 MoE 路由的消息不规则、大小可变，NCCL 效率低。脚本里 `NVSHMEM_DISABLE_NCCL=1` 禁止 DeepEP 的 NVSHMEM 降级回 NCCL，强制走纯 RDMA。

#### How to Inspect Local IB Devices

```bash
ls /sys/class/infiniband/    # 列出所有 IB 设备，如 mlx5_0 mlx5_1 ...
ibstat                       # 查每张卡状态（Active 才可用）
ibdev2netdev                 # 查 IB 设备与网口对应
nvidia-smi topo -m           # 查 GPU 与 IB 网卡的亲和性
```

脚本里的 `mlx5_100`–`mlx5_107` 编号高，是因为该集群有大量虚拟网络接口占用了低编号。部署时先用 `ls /sys/class/infiniband/` 看真实名称，不要猜。

### Compute and Compilation: Kernel, Eager, CUDA Graph, JIT

decode 阶段 GPU 利用率可能只有约 33%，不是 GPU 算不动，是它在等 CPU。要看懂 `cuda_graph_max_bs`、`SGLANG_JIT_DEEPGEMM_PRECOMPILE` 这些参数在优化什么，先拆开一次 forward 到底由什么组成。

#### A Forward Pass Is Thousands of Kernels

矩阵乘、softmax、layernorm、逐元素加法，每一个都是一个独立的 **kernel**（GPU 上一段可执行计算）。一个 transformer 层几十个 kernel，几十层叠起来，一次 forward 到 2000 个 kernel 量级。

每个 kernel 都要 **launch**：CPU 准备参数、告诉 GPU 用多少线程、从哪读、往哪写，GPU 才开始算。线程数由数据量决定——batch=8 需要 800 个线程，batch=16 需要 1600 个。每次 launch 有约 5 微秒的固定 CPU 开销。

#### Eager Mode Makes the CPU the Bottleneck

PyTorch 默认走 eager：CPU 一个接一个 launch，launch 完一个等 GPU 算，再 launch 下一个。2000 个 kernel、每个 5 微秒，CPU launch 总开销 10 ms。大 batch prefill 的 GPU 计算要 100 ms，这 10 ms 可忽略；但 decode 每步只生成一个 token，GPU 实际算可能只要 5 ms：

```
GPU 实际算：5 ms
CPU launch：10 ms   ← 比 GPU 算的还久
结果：GPU 大部分时间在等 CPU 发下一个 launch
```

瓶颈不在 GPU，在 CPU。这不是 eager 写得差，是它的固有代价：launch 次数等于 kernel 数。提速两条路——减少 launch 次数（CUDA Graph），或减少 kernel 首次编译开销（JIT 预编译）。

#### CUDA Graph: Record Once, Replay to Eliminate Launch Overhead

CUDA Graph 把整串 kernel 序列录制成一张图，之后一次 launch 重放整张图：

```
录制（capture，启动时一次）：记录 2000 个 kernel 的顺序、参数、内存地址
重放（replay，每步推理）：CPU 只发 1 次 launch → GPU 自己跑完 2000 个
CPU launch 开销：10 ms → 5μs，GPU 利用率从 33% 拉到接近 100%
```

CUDA Graph 不是万能加速开关。它有一个硬约束：录制时把线程数和内存地址焊死了，重放必须用完全相同的形状。batch 变了，形状变了，这张图作废。这也是它区别于 TensorFlow 静态图的地方——TF 静态图记的是"要做什么"的逻辑蓝图，batch 维度可待定，执行时按真实值再算线程数；CUDA Graph 记的是"上次具体怎么做的"物理录像，一切都是定值。

SGLang 的应对：为多个 batch size 各录一张图（bs=1,2,4,8,12），共享固定预分配 buffer 保证地址不变；运行时把实际 batch 向上 padding 到最近的已录档位（bs=7 塞进 bs=8 的图），超过 `cuda_graph_max_bs` 才退回 eager。分工上，eager 灵活但慢，适合形状多变的 prefill；CUDA Graph 快但形状固定，适合形状稳定的 decode。

#### JIT Compiles Per-Kernel Machine Code

JIT（Just-In-Time）编译的对象是单个 kernel 的机器码，不是整个模型，也不是 kernel 的调用顺序。它和 CUDA Graph 是两个层次：JIT 管每个 kernel 怎么造，CUDA Graph 管 2000 个 kernel 怎么串起来一次发出去。

一个 kernel 的最优机器码取决于三个只有运行时才确定的信息：矩阵形状（M×K×N 决定分块策略）、GPU 架构（Hopper SM90 与 Blackwell SM100 指令集不同）、数据类型（FP8 与 BF16 走不同路径）。三者组合爆炸，不可能提前全编译。所以 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、Triton 这类高性能 kernel 库写成模板，第一次遇到某组合才编译、然后缓存，下次同组合直接复用。

JIT 的代价只在第一次：编译一个新形状要停下几秒。问题是训练和 prefill 里矩阵形状一直变，每个新形状都是一次新编译，训练开头几十步可能全耗在编译上。解法是 pad 分桶加缓存：把千变万化的形状向上取整到少数固定档位（如都 pad 到 128 的倍数），几千种形状收敛成十几个桶；启动时预编译一遍（warmup），之后全程命中缓存。`SGLANG_JIT_DEEPGEMM_PRECOMPILE=true` 就是这个思路——把编译集中到启动，别散落在运行时拖慢每一步。

#### Eager vs. CUDA Graph and JIT Are Orthogonal

常见误解是把 eager 和 JIT 当成对立的两条路。它们互不相干：

```
维度一：kernel 怎么发射？      eager（逐个 launch） vs CUDA Graph（录制重放）
维度二：kernel 机器码哪来的？  JIT（运行时编译） vs AOT（提前编译好）
```

| | JIT kernel | AOT kernel |
|---|---|---|
| **eager** | 最常见组合 | 可以 |
| **CUDA Graph** | 录制时不能 JIT，必须先预编译 | 可以 |

eager 加 JIT 天然兼容：eager 真在执行 kernel，第一次碰到新形状时 JIT 就地编译、缓存、再 launch。所以 prefill 走 eager 不代表不编译——恰恰相反，prefill 形状多变，是 JIT 触发最频繁的地方。真正与 JIT 冲突的是 CUDA Graph 的录制阶段：录制时 CUDA 进入只记录不真跑的模式，JIT 需要真执行，录制模式下做不了编译。这就是 `SGLANG_JIT_DEEPGEMM_PRECOMPILE=true` 必须先于 CUDA Graph 录制的原因。

#### Lifecycle: Compile and Record at Startup, Only Replay at Runtime

JIT 和 CUDA Graph 不在同一次 forward 里，分属两个阶段，中间有一条硬依赖：

```
启动阶段（只做一次）
  1. 加载模型权重
  2. 分配 KV cache 静态池
  3. JIT 预编译：把常用形状的 kernel 全部编译好、缓存
  4. CUDA Graph 录制：对每个 batch 档位跑一次假 forward，录成图

运行阶段（每步反复做）
  5. decode：把实际 batch padding 到最近档位 → 重放对应的图 → 出 token
  6. prefill：形状多变，走 eager 或 piecewise graph
```

第 3 步必须在第 4 步之前，因为录制模式下做不了 JIT。先 JIT 把每块砖造好，再用 CUDA Graph 把砖码成墙记下来，运行时整墙重放。一次 decode 的运行时流程里既没有 JIT 也没有录制，只有重放——这正是它快的原因。

### THD Data Layout and Multi-Request Batching

SGLang 一次 forward 同时处理多个请求，靠的是 THD 数据布局。理解它才懂 batch、`cu_seqlens`、memory-bound 从哪来。

#### THD Is Packed and Padding-Free, Not BSHD

SGLang 是纯推理引擎，没有 backward。它的 forward 全程用 THD（packed/varlen）格式——把所有请求的 token 首尾拼成一条 1D 序列，用 `cu_seqlens`（累积序列长度）标记边界：

```
BSHD（padding 到最长，浪费）:        THD（SGLang 用这个，无浪费）:
  seq0: [t0 t1 t2 PAD PAD]           input_ids  = [t0 t1 t2 | t3 t4 | t5 t6 t7 t8]
  seq1: [t3 t4 PAD PAD PAD]          cu_seqlens = [0, 3, 5, 9]
  seq2: [t5 t6 t7 t8 PAD]
```

推理请求长度差异极大（有的 100 token，有的 60000），BSHD padding 到最长会把显存和算力浪费在 PAD 上。THD 只算真实 token，是变长请求高吞吐的刚需。它对应 Megatron THD 的 `cu_seqlens` / `PackedSeqParams`，区别是 SGLang 只有 forward——THD 反向只发生在训练侧。

贯穿全节的关键洞察：一次 forward 的层分两类。线性层、FFN、LayerNorm 逐 token 独立计算，多请求 token 混在一起做一把大矩阵乘也不串味，完全不管边界（这也是 THD 高效的原因，一次大矩阵乘顶过去多次小的）。只有 attention 关心边界，必须防止请求 A 的 token 看到 B。所以下面 prefill / decode 各讲两件事：attention 怎么隔离、编译怎么应对形状。

#### Prefill: Block-Diagonal Square Matrix, Compilation Hotspot

prefill 的 `input_ids` 装每个请求的整个 prompt，多请求打包成一条。Q 和 K/V 都来自这条打包序列，attention 形成 M×M 方阵，`cu_seqlens` 把它切成互不干扰的对角块：

```
            key →   A0 A1 A2 │ B0 B1 │ C0 C1 C2 C3
 query ↓
    A0              ✓  ·  ·  │ ·  ·  │ ·  ·  ·  ·    ┐
    A1              ✓  ✓  ·  │ ·  ·  │ ·  ·  ·  ·    │ A 只看 A（因果，只看前面）
    A2              ✓  ✓  ✓  │ ·  ·  │ ·  ·  ·  ·    ┘
    ────────────────────────┼───────┼────────────
    B0              ·  ·  ·  │ ✓  ·  │ ·  ·  ·  ·    ┐ B 只看 B
    B1              ·  ·  ·  │ ✓  ✓  │ ·  ·  ·  ·    ┘
    ────────────────────────┼───────┼────────────
    C0              ·  ·  ·  │ ·  ·  │ ✓  ·  ·  ·    ┐
    C1              ·  ·  ·  │ ·  ·  │ ✓  ✓  ·  ·    │ C 只看 C
    C2              ·  ·  ·  │ ·  ·  │ ✓  ✓  ✓  ·    │
    C3              ·  ·  ·  │ ·  ·  │ ✓  ✓  ✓  ✓    ┘

✓=参与 attention  ·=被 mask。三个下三角块贴在对角线上，块外全是 ·（跨请求全屏蔽）。
cu_seqlens=[0,3,5,9] 就是这三块的边界。
```

编译上，prefill 的 M（所有 prompt 长度之和）变化范围极大：CUDA Graph 难用（没法为每个 M 录图，走 eager 或 piecewise graph），JIT 频繁触发（每个新 M 都是新形状）。所以 prefill 是编译重灾区，靠 pad 分桶把 M 收敛到少数档位、warmup 编译完复用，`SGLANG_JIT_DEEPGEMM_PRECOMPILE=true` 主要就是为这些形状预编译。

#### Decode: Short-Wide Matrix, CUDA Graph Territory

decode 的 `input_ids` 每个请求只装 1 个新 token，历史 KV 不在 input_ids 里、在分页 KV cache 中。新 token 当 query（只有 B 个），历史 KV 当 key/value（从 paged cache 按 page table 读），矩阵从方阵变成"B 行 query × 各自历史 KV"的矮宽形状：

```
input_ids = [A_new  B_new  C_new]     ← 只有 3 个新 token，M = B = 3
seq_lens  = [6, 4, 8]                 ← A/B/C 各自历史 KV 长度
        A的KV(6)      B的KV(4)    C的KV(8)
  A_new [✓✓✓✓✓✓]    [      ]    [        ]   ← 各请求用自己的 seq_lens/page 表
  B_new [      ]    [✓✓✓✓]      [        ]      只读自己的 KV，互不干扰
  C_new [      ]    [    ]      [✓✓✓✓✓✓✓✓]
```

decode 是 memory-bound：每个新 token 算得极少（1 个 query 向量），却要读整个历史 KV（可能几万个）。算得少读得多，瓶颈在 KV cache 读带宽。这是 FP8 KV cache 有效的根本原因——KV 减半，读带宽减半，decode 提速。

编译上，decode 的 M = B（每请求 1 个新 token），请求数档位有限（1,2,4,8,12），形状稳定：CUDA Graph 好用（为每档预录一张图重放），JIT 基本不触发。但 CUDA Graph 仍必须按 B 录制——`cu_seqlens` / `kv_indptr` / page 表这些元数据数组的长度随请求数 B 变，B 变了图不能复用。这就是 `cuda_graph_max_bs` 按 batch 命名的原因。

> 一个易混点：THD 不 padding 的是序列长度维度；CUDA Graph padding 的是请求数维度（补到已录档位）。两者作用在不同维度，不矛盾。

---

## Request Scheduling: Two Layers of Load Balancing

### Principle

一个请求从进入集群到落到某张 GPU，经过**两层各自独立的调度**，别混：

```
请求 → [第一层] Router 跨副本路由        --policy         cache_aware / consistent_hashing / round_robin
     → [第二层] 引擎内 DP rank 分派      load_balance_method   follow_bootstrap_room / round_robin / ...
     → 具体 DP rank 的 scheduler
```

- **第一层 Router（`--policy`）**：SGLang Model Gateway 决定请求发给哪个 engine 副本。256 卡切成多副本时，它管"哪个副本"。
- **第二层 `load_balance_method`**（[server_args.py:905](python/sglang/srt/server_args.py#L905)，合法值 `auto`/`round_robin`/`follow_bootstrap_room`/`total_requests`/`total_tokens`）：请求进了某个 engine 后，[DataParallelController](python/sglang/srt/managers/data_parallel_controller.py#L128) 决定它落到该 engine 内 64 个 DP rank 的哪一个。

**本脚本设的是第二层**（prefill `follow_bootstrap_room`、decode `round_robin`），不是第一层的 `--policy`。这俩不是同一个东西，也不是替代关系——第一层选副本、第二层选 rank，一个请求两层都要过。cache_aware / consistent_hashing 是第一层的选项，跟脚本这两个值不在一个层次。

### Parameter Setting

**prefill 侧 `load_balance_method = follow_bootstrap_room`**。核心逻辑 [data_parallel_controller.py:635](python/sglang/srt/managers/data_parallel_controller.py#L635)：

```python
target_rank = req.bootstrap_room % len(self.workers)   # 按 bootstrap_room 取模选 DP rank
```

`bootstrap_room` 是每个请求的唯一 id，PD 两侧共享。prefill 和 decode 都用它取模，同一请求就落到**编号一致的 DP rank**——prefill 的 KV 直接点对点 RDMA 写到 decode 对应 rank，无需额外路由层查表。

**decode 侧 `load_balance_method = round_robin`**。[data_parallel_controller.py:612](python/sglang/srt/managers/data_parallel_controller.py#L612)，一个计数器轮流分派。decode 不需要 KV 落点可预测（KV 是 prefill 传来的、已到位），只求均匀填满各 rank。

### QA

**为什么 prefill 不用默认的 cache_aware / consistent_hashing**。因为这俩根本不是同一层的选项——它们是 Router 的 `--policy`（第一层，选副本），而 `load_balance_method` 是引擎内 DP rank 分派（第二层）。两层解耦，脚本在第二层显式设值，不影响第一层用什么 policy。

**为什么第二层 prefill 偏要 follow_bootstrap_room 而非 round_robin**。PD 分离下 KV 要从 prefill 传到 decode。若 prefill 用 round_robin 随机落 rank，decode 侧不知道 KV 在 prefill 的哪个 rank，得靠额外路由表查、再跨 rank 搬运。用 `bootstrap_room % dp_size` 让两侧落点用同一公式算出、天然一致，KV 传输退化成点对点 RDMA write，省掉查表和跨 rank 搬运。decode 侧没有"要往哪传"的约束，所以回到 round_robin 求均衡。

**第一层那个 --policy 谁在设**。不在这份训练脚本里，在拉起 Router 的地方（slime 默认 `cache_aware`）。它和本节讲的第二层是两码事；第一层的 cache_aware / consistent_hashing 行为、以及 kwaienv 沙箱下 session 亲和为何失效，见 [PD Disaggregation 的 QA](#qa-1)。

---

## Config at a Glance

脚本暴露两套配置：PD 分离（YAML `SGLANG_CONFIG_FILE`，1 prefill + 3 decode engine，各 64 GPU）与 colocate（命令行 `SGLANG_ARGS`，每 engine 64 GPU）。`--sglang-config` 存在时以 YAML 为准，colocate 参数作 fallback。三者并排看，加粗值是该列独有的差异：

| 参数 | PD prefill | PD decode | colocate |
|---|---|---|---|
| **并行** | | | |
| dp_size / ep_size | 64 / 64 | 64 / 64 | 64 / 64 |
| enable_dp_attention | ✓ | ✓ | ✓ |
| enable_dp_lm_head | ✓ | ✓ | ✓ |
| moe_dense_tp_size | 1 | 1 | 1 |
| **MoE 通信** | | | |
| moe_a2a_backend | deepep | deepep | deepep |
| deepep_mode | auto | **low_latency** | auto |
| moe_runner_backend | 默认 | **deep_gemm** | 默认 |
| **Attention / KV** | | | |
| attention_backend | nsa | nsa | nsa |
| nsa_prefill / nsa_decode | flashmla_sparse / flashmla_kv | 同左 | 同左 |
| kv_cache_dtype | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 |
| page_size | 64 | 64 | 64 |
| mem_fraction_static | — | — | 0.70 |
| **PD 传输** | | | |
| transfer_backend | mooncake | mooncake | mooncake |
| load_balance_method | **follow_bootstrap_room** | **round_robin** | round_robin |
| **调度容量** | | | |
| chunked_prefill_size | **131072** | — | 默认 |
| max_running_requests | 512 | **768** | 512 |
| cuda_graph_max_bs | 默认 | **12** | **8** |
| disable_overlap_schedule | 否 | **是** | 是 |
| watchdog_timeout | 3600 | 3600 | 3600 |
| **投机解码（EAGLE + MTP）** | | | |
| num_steps / topk / draft_tokens | 4 / 1 / 5 | 4 / 1 / 5 | 4 / 1 / 5 |
| draft_attention_backend | nsa | nsa | nsa |

加粗的 7 个参数是 PD 两侧 / colocate 的全部差异点，其余参数三列一致（模型与硬件绑定）。差异根因见后文 [PD vs Colocate 参数差异汇总](#pd-vs-colocate-参数差异汇总)。

<details>
<summary>原始配置片段（可复制）</summary>

```yaml
# PD 分离 SGLANG_CONFIG_FILE
prefill overrides: dp_size=64 ep_size=64 enable_dp_attention=true enable_dp_lm_head=true
                   moe_dense_tp_size=1 load_balance_method=follow_bootstrap_room
                   chunked_prefill_size=131072 max_running_requests=512 deepep_mode=auto
decode  overrides: dp_size=64 ep_size=64 enable_dp_attention=true enable_dp_lm_head=true
                   moe_dense_tp_size=1 load_balance_method=round_robin max_running_requests=768
                   cuda_graph_max_bs=12 deepep_mode=low_latency
                   moe_runner_backend=deep_gemm disable_overlap_schedule=true
```

```bash
# colocate SGLANG_ARGS
--sglang-mem-fraction-static 0.70 --sglang-ep-size 64 --sglang-dp-size 64
--sglang-enable-dp-attention --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1
--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto
--sglang-page-size 64 --sglang-kv-cache-dtype fp8_e4m3
--sglang-attention-backend nsa --sglang-nsa-prefill-backend flashmla_sparse --sglang-nsa-decode-backend flashmla_kv
--sglang-cuda-graph-max-bs 8 --sglang-disable-overlap-schedule
--sglang-max-running-requests 512 --sglang-watchdog-timeout 3600
--sglang-disaggregation-transfer-backend mooncake
--sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps 4
--sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens 5
--sglang-speculative-draft-attention-backend nsa
```

</details>

---

## Parallelism and MoE Communication

### Principle

并行决定权重怎么切到多卡，通信决定切开后怎么把数据在卡间搬回来。MoE 模型这两件事强耦合。

MoE 模型（本脚本的模型有 256 个专家）分布式下每张 GPU 只持有部分专家（256÷64=4 个），每个 token 被 router 路由到对应专家所在的 GPU，需要全局 token 重排。一次 MoE forward 有两次 all-to-all：Dispatch（token 发到 expert GPU）→ 本地矩阵乘 → Combine（结果返回原始 GPU）：

```
Dispatch 之后：
GPU0 收到所有分配给专家0-3 的 token（来自全部 64 张卡）
GPU1 收到所有分配给专家4-7 的 token（来自全部 64 张卡）
```

这是多机 MoE 的主要通信开销。[DeepEP](https://github.com/deepseek-ai/DeepEP)（DeepSeek 开源）用 RDMA 原语优化这个不规则消息模式，比 NCCL 的通用 all-to-all 高效。GEMM 后端用 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)（DeepSeek 开源）。DP-Attention / EP / TP 的体系讲解见 `sglang-features.md` 与 `2-sglang.md`，此处只讲参数。

### Parameter Setting

**`enable_dp_lm_head`**：LM head（hidden 投影到 vocab 的矩阵乘）按 DP-Attention 组本地计算，不做跨组 all-gather（[logits_processor.py:340](python/sglang/srt/layers/logits_processor.py#L340)）。前提是先开 `--enable-dp-attention`。大词表模型下，dp_size=64 时每 rank 只算 1/64 batch 的 vocab logits，消除的 all-gather 通信不可忽视，所以开。

**`moe_dense_tp_size = 1`**：控制 dense 层（非 MoE FFN、attention 后的 dense projection）的 TP 切分宽度（[server_args.py:7053](python/sglang/srt/server_args.py#L7053) 只允许 `None`/`1`/`tp_size`）。设 1 = dense 层不做 TP，每卡各持完整 dense 权重独立算。为什么：tp_size=64 下若切 64 份，单份 GEMM 维度过小、kernel 利用率崩塌；dense 层参数量相对 expert 小，复制 64 份显存代价可接受，换来零通信。三处一致。

**`moe_a2a_backend = deepep`**：all-to-all 用 DeepEP。三处一致。

**`deepep_mode`**：DeepEP kernel 模式（[server_args.py:1925](python/sglang/srt/server_args.py#L1925)）。`normal` 面向 prefill/大 batch、吞吐优先、不兼容 CUDA graph；`low_latency` 面向 decode/小 batch、启动开销极低、兼容 CUDA graph；`auto` 按 batch 类型动态切。

| 部署 | 取值 | 为什么 |
|---|---|---|
| PD prefill | `auto` | 偶尔夹小 batch 时能切 low_latency |
| PD decode | `low_latency` | 纯 decode，钉死省判断开销，保证 CUDA graph 全程可用 |
| colocate | `auto` | 单引擎混跑，只能动态切 |

**`moe_runner_backend = deep_gemm`（仅 decode）**：MoE GEMM 后端（[server_args.py:1429](python/sglang/srt/server_args.py#L1429)），JIT 编译、针对小 M 优化。只在 decode 设——decode 是 memory-bound 小 M GEMM，收益明显；prefill 大 M 用默认 auto。

**环境变量**：
- `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = 64`（[environ.py:665](python/sglang/srt/environ.py#L665)，默认 128）：low_latency kernel 每 EP rank 最多 dispatch 的 token 数。decode 并发 768、ep_size 64，单 rank 期望 12，设 64 是保守上限，同时比默认省一半预分配显存。
- `SGLANG_JIT_DEEPGEMM_PRECOMPILE = true`（[environ.py:648](python/sglang/srt/environ.py#L648)）：启动预编译 DeepGEMM，避免第一批 decode 触发 JIT 延迟撞 watchdog（原理见 Basic 计算与编译节）。
- `NVSHMEM_DISABLE_NCCL = 1`：禁止 DeepEP 的 NVSHMEM 降级回 NCCL，强制纯 RDMA。属 NVSHMEM 库，非 sglang 变量。

### QA

**为什么 all-to-all 不用 NCCL**：NCCL 的 all-to-all 面向对称固定大小消息，而 MoE 路由每步每卡发往各目标的 token 数不定、消息不规则，NCCL 效率低。DeepEP 用 RDMA 原语专门优化不规则 dispatch/combine。

**deep_gemm 为什么要预编译**：它是 JIT kernel，第一次遇到某矩阵形状要现场编译（几秒）。不预编译则第一批 decode 请求触发编译、首个 forward 变慢，可能撞 watchdog_timeout 崩溃。预编译把这步挪到启动。

**ep_size 为什么锁 64**：DeepEP 要求 `ep_size = tp_size`，脚本 moe_dense_tp_size=1、dp_size=64、每 DP rank 1 卡、MoE TP=1，所有 64 卡按 EP 维度排列。这也直接决定了每台 engine 必须 64 GPU（见 PD 分离节的硬件分配）。

---

## PD Disaggregation

### Principle

prefill 和 decode 混在一个引擎里会互相拖累：prefill batch 抢占正在跑的 decode batch，decode 延迟抖动。PD 分离把两阶段拆到独立引擎，各自优化。DSA 稀疏 attention 与 PD 强相关——prefill 和 decode 走不同 kernel，正是 PD 能各自优化的例子——一并放这节。PD 分离的体系讲解见 `sglang-features.md`。

硬件分配 1P + 3D × 64 GPU 的由来：每台 engine 锁 64 GPU（ep_size=64，见上一节 QA）。256 ÷ 64 = 4 台，选项只有 1P+3D、2P+2D、3P+1D。选 1P+3D 因为 prefill compute-bound 完成快（一次 forward），decode memory-bound 要逐 token 生成（最多 65536 步），时间比 ≈ 65536:1，瓶颈在 decode——1 台 prefill 512 并发能持续喂满 3 台 decode 合计 2304 并发，2P+2D 则 prefill 空等、decode 积压。

DSA（DeepSeek Sparse Attention）只选一部分 KV 参与 attention 计算，kernel 来自 [FlashMLA](https://github.com/deepseek-ai/FlashMLA)（DeepSeek 开源），机制见 [NSA 论文](https://arxiv.org/abs/2502.11089)。prefill 和 decode 对稀疏 attention 需求不同，所以拆两个 kernel。

KV 跨引擎传输走 Mooncake：以 KV page 为粒度，RDMA one-sided write 把 prefill 生成的 KV 直接写入 decode 侧 allocator，不经 CPU（[mooncake/conn.py](python/sglang/srt/disaggregation/mooncake/conn.py)）。

### Parameter Setting

**`attention_backend = nsa`**：`nsa` 是 `dsa` 的废弃别名，等价（[server_args.py:200](python/sglang/srt/server_args.py#L200) 注释 `# Deprecated alias for "dsa"`）。实现在 [dsa_backend.py](python/sglang/srt/layers/attention/dsa_backend.py)，追代码以此为准。三处一致。

**`nsa_prefill_backend = flashmla_sparse` / `nsa_decode_backend = flashmla_kv`**：prefill 长 prompt 全序列，sparse index 过滤大量 KV block、跳过不参与计算的 block 做大矩阵乘；decode 每步 1–12 个新 token，把 FP8 MLA 压缩 KV 展开成标准 MHA，展开后 query head 数由 [dsa_backend.py:385](python/sglang/srt/layers/attention/dsa_backend.py#L385) 决定（≤64 补到 64，≤128 补到 128）。显式指定两个 backend 覆盖 [dsa_backend.py:2825](python/sglang/srt/layers/attention/dsa_backend.py#L2825) 的自动选择，避免运行时判断抖动。

**`INDEXER_ROPE_NEOX_STYLE = 0`（外部环境变量）**：控制 DSA indexer 的 RoPE 风格，属外部 indexer 库、不在 sglang 源码。设 0 = 非 NeoX 风格，须与模型权重的 RoPE 约定一致，改错会让 sparse index 选错 KV block、attention 输出异常。

**`disaggregation_transfer_backend = mooncake`**：KV 传输后端（原理见 Principle）。

**`disaggregation_ib_device`**：Mooncake 用哪些 IB 网卡做 RDMA。不填自动探测（可能选错）。校验在 [server_args.py:5999](python/sglang/srt/server_args.py#L5999)，四种格式：

| 格式 | 示例 | 适用 |
|------|------|------|
| 单设备 | `mlx5_0` | 每节点 1 张 IB 卡 |
| 逗号分隔（共享） | `mlx5_0,mlx5_1,mlx5_2` | 同构集群，所有 GPU 共用 |
| per-GPU JSON | `{"0":"mlx5_0","1":"mlx5_1"}` | 拓扑不对称，精细控制 |
| JSON 文件路径 | `/path/to/ib_map.json` | 多节点配置不同 |

本脚本用格式二：8 张卡共享、聚合 1.6 Tb/s。

**`load_balance_method`**：prefill 用 `follow_bootstrap_room`，decode 用 `round_robin`（[server_args.py:905](python/sglang/srt/server_args.py#L905)）。prefill 侧核心逻辑 [decode.py:562](python/sglang/srt/disaggregation/decode.py#L562)：`return req.bootstrap_room % prefill_info.dp_size`——同一请求在 prefill 和 decode 落到编号一致的 DP rank，KV 传输变成点对点 RDMA write。decode 侧 round_robin 均匀分配到 3 个 engine。两侧不同：prefill 要 KV 落点可预测（决定往哪个 decode rank 传），decode 只关心负载均衡。

**`chunked_prefill_size = 131072`（仅 prefill）**：每个 chunk 最大 token 数（[server_args.py:698](python/sglang/srt/server_args.py#L698)，`-1` 禁用）。prefill 专机没有 decode 抢时隙，可用大 chunk 一次吃进长 prompt 最大化吞吐；colocate 要和 decode 共享时间轴，需小 chunk 频繁让路，脚本里 colocate 不设、走默认。

### QA

**PD 分离能带来多少性能提升**（无该脚本直接 A/B 数据，标注 原理确定 / 公开数据 / 估算）。colocate 慢在两处结构性损耗：prefill batch 抢占 decode，decode TBT 抖动、P99 远高于 P50（原理确定）；DP-Attention 64 rank 必须同步 step，跑 prefill 的和跑 decode 的计算量差异大、快的等慢的（原理确定）。

| 指标 | 改善幅度 | 依据 |
|------|---------|------|
| Decode TBT P50 | 2–4× 降低 | 估算，取决于 prefill 压力 |
| Decode TBT P99（抖动） | 5–10× 降低 | 原理确定 |
| 总 tokens/s 吞吐 | 20–40% 提升 | 公开数据（DeepSeek-V3 场景） |
| TTFT | +<1 ms（略差） | RDMA KV 传输开销 |
| RL rollout 轮次时长 | 缩短 30–50% | 估算 |

RL 训练循环串行，整轮等待 = 最慢轨迹（P99）完成时间。colocate decode 抖动大 → P99 拖长 → 训练侧空等。PD 分离消除抖动，steps/hour 提升比 tokens/s 更显著。收益小的场景：prompt < 200 token 且 QPS 低，或 response < 1K token。

**kwaienv 沙箱场景下多副本怎么负载均衡**。slime 把 batch 请求发给 kwaienv，kwaienv 在沙箱内起 ProxyServer 代理 LLM 调用，sglang 侧起 router 做多副本负载均衡。请求链路：

```
slime rollout → _submit_to_agent_cluster(sample)
  → ModelConfig(base_url=router/v1, api_key="dummy_key")
    → kwaienv 沙箱内 ProxyServer
      → POST /v1/chat/completions → sglang router
         Authorization: Bearer dummy_key   ← 硬编码，无 X-SMG-Routing-Key
      → router 按策略分发 → 副本1 / ... / 副本K
```

session_id 没有参与路由：[kwai_env_system.py:321](examples/kat/kwai_env_system.py#L321) 的 `api_key="dummy_key"` 硬编码，全部 sample 共用；[proxy_server.py:778](kwaienv/utils/proxy_server.py#L778) 转发时写 `Bearer {self.api_key}`，所有请求都是 `dummy_key`。结果里的 `session_id: f"agent_{eval_id}"` 只是标签，不影响路由。

| 策略 | 256 卡多副本实际行为 | 建议 |
|------|---------------------|------|
| `cache_aware`（默认） | 无 routing key，按缓存命中率+负载分散 | **用这个** |
| `consistent_hashing` | 所有请求 routing key 都是 `dummy_key` → 全打同一副本 | 绝对不能用 |
| `round_robin` | 纯轮询 | 追求绝对均衡时可选 |

若要真正 session 亲和：把 `api_key=f"session_{eval_id}"` 注入，再配 `--router-policy consistent_hashing`，同一 sample 的多轮请求落同一副本、prefix cache 完整复用（需改 kwaienv 侧代码）。

<a id="pd-vs-colocate-参数差异汇总"></a>
**PD 和 colocate 到底差哪几个参数**。两套配置只在"prefill 和 decode 是否共享调度时间轴"上分叉，差异全部由此派生。

| 参数 | PD prefill | PD decode | colocate | 分叉根因 |
|---|---|---|---|---|
| deepep_mode | auto | low_latency | auto | decode 专机才能钉死 low_latency |
| moe_runner_backend | 默认 | deep_gemm | 默认 | decode 小 M GEMM 优化 |
| load_balance_method | follow_bootstrap_room | round_robin | round_robin | prefill 要 KV 落点可预测 |
| chunked_prefill_size | 131072 | — | 默认 | prefill 专机可用大 chunk |
| max_running_requests | 512 | 768 | 512 | decode memory-bound 可高并发 |
| cuda_graph_max_bs | 默认 | 12 | 8 | decode 单 rank batch = 768/64 |
| disable_overlap_schedule | 否 | 是 | 是 | decode/colocate 跑投机解码 |

不分叉的参数（模型/硬件绑定，抄就行）：`dp_size=ep_size=64`、`enable_dp_attention`、`enable_dp_lm_head`、`moe_dense_tp_size=1`、`page_size=64`、`kv_cache_dtype=fp8_e4m3`、`attention_backend=nsa` 及两个 sub-backend、EAGLE 四参数、`watchdog_timeout=3600`。调参先抄不分叉的一组，再按部署形态调分叉的 7 个。

**若改成 ep=tp=32 怎么分**。每台 engine 缩到 32 GPU、总 8 台，decode:prefill 比例不变（还是 3:1）→ 2P+6D。比例由 workload 决定（decode 步数 >> prefill），与 ep 大小无关；ep=32 只是每台更小、台数翻倍。

**256 卡 / 32 节点为什么 ib-device 只写 8 个**。这是每节点本地配置，不是全集群。每节点只声明本机 8 张出口网卡，节点间连接由 Mooncake 元数据服务器协调，不需列出对端网卡；同构集群所有节点 IB 编号相同，同一份参数下发全部 32 台。不写也行（自动探测），但有复杂虚拟接口时可能选错，建议显式写。

---

## CUDA Graph and Scheduling

### Principle

CUDA Graph 消除 kernel launch 开销，原理见 Basic 计算与编译节。这里三个参数都是调度容量的上限——`max_running_requests` 限并发请求数，`cuda_graph_max_bs` 限能命中 graph 的 batch，`watchdog_timeout` 限单步超时。三者取值都由 decode/prefill 的 memory-bound vs compute-bound 差异派生。

### Parameter Setting

**`cuda_graph_max_bs`：decode 12 / colocate 8**（[server_args.py:1514](python/sglang/srt/server_args.py#L1514)）：CUDA graph 捕获的最大 batch size，运行时 batch ≤ 此值才命中 graph。decode 设 12 因为 DP-Attention 下每 rank batch = 总并发/dp_size = 768/64 = 12，正好对齐；colocate 混跑单 rank 有效 batch 更小，8 够用且省 graph 显存。与 deepep_mode 耦合：decode 钉死 low_latency 才能让此值生效，走 normal 则 CUDA graph 被禁用、此值无意义。

**`max_running_requests`：prefill 512 / decode 768 / colocate 512**（[server_args.py:678](python/sglang/srt/server_args.py#L678)，默认 None 自动算）：engine 内并发运行请求上限。不是 batch size，是同时在跑的请求数（每步实际 batch 由调度器按 KV 剩余页动态定）。prefill compute-bound、单请求算力占用大，并发不宜过高；decode memory-bound、单请求算力小，多并发填满 batch 喂饱 low_latency kernel；colocate 取保守值避免大 prefill 挤爆 KV pool。decode 比 prefill 高 50% 正是两种 bound 的并发差异。

**`watchdog_timeout = 3600`**（[server_args.py:1015](python/sglang/srt/server_args.py#L1015)，默认 300 秒）：单 forward batch 超时则 server 崩溃防挂死。RL rollout 拉到 3600 因为 prefill 长 prompt（131072 chunk）+ 大 batch 单步可能远超 300 秒，DeepEP 首次 dispatch、deep_gemm JIT 编译也会拖长首批 forward。默认 300 是为在线 serving 设的。

### QA

**max_running_requests 是最大 batch size 吗**。不是。它是同时在跑的请求数上限，每步实际 batch（token 数）由调度器按 KV cache 剩余页动态决定：实际并发 = min(max_running_requests, KV 页够放几个请求)。

**1024 个请求、max 256，会 OOM 吗**。不会。KV cache 是启动时按 mem_fraction_static 预分配的固定池，请求只是从池里"借"page；调度器放行前检查剩余页够不够，不够就排队。超出 256 的请求排队，靠 continuous batching 逐个补进，不影响吞吐（前提是 batch 保持充满）。真正的 OOM 风险在 prefill 大 chunk 的激活内存，靠 mem_fraction_static 留的余量兜。

---

## KV Cache

### Principle

KV cache 是 decode 阶段的主要显存开销，也是 decode memory-bound 的瓶颈所在（原理见 THD 数据布局节）。降低它的存储精度直接减少读带宽，是 decode 提速最直接的杠杆。本脚本用 FP8 存 KV，配合 flashmla_kv kernel 直接读 FP8、不做类型转换。

### Parameter Setting

**`kv_cache_dtype = fp8_e4m3`**（[server_args.py:601](python/sglang/srt/server_args.py#L601)，choices `fp8_e5m2`/`fp8_e4m3`/`bf16`/`fp4_e2m1`，默认与计算精度一致通常 BF16）：fp8_e4m3（4 位指数 3 位尾数）比 e5m2 精度更高、动态范围 ±240，适合 KV。flashmla_kv decode kernel 直接读 FP8，节省显存和读带宽。

**`page_size = 64`**（[server_args.py:775](python/sglang/srt/server_args.py#L775)）：KV cache 一个 page 含多少 token。64 是 flashmla kernel 的 page alignment 要求，也是 Mooncake KV 传输的基本粒度。改成非对齐值会让 flashmla 走慢路径或报错。

### QA

**FP8 KV 会不会影响推理精度**。有影响但很小。BF16 → FP8 E4M3 丢约 4 bit 精度，但 KV 值分布平滑（attention 天然抑制极端值），量化误差不显著，DeepSeek-V3、vLLM、SGLang 生产部署已大规模验证。

**会不会加大训推不一致**。会，但量级小、可管理。FP8 KV 的误差叠在本就存在的训推不一致（batch size 不同、kernel 不同）之上：

```
训练侧：BF16 权重 + BF16 激活 → 重算 log-prob
推理侧：BF16 权重 + FP8 KV（量化误差）→ log-prob 略不同
```

本脚本里三个理由使这不成问题：`--use-tis --tis-clip-low 0.5 --tis-clip 2.0` 通过 importance sampling 修正 off-policy 比率，FP8 偏差在容忍范围内；误差量级约 1e-3，远小于 RL 优势估计噪声；用 BF16 KV 显存 2×、max_response_len=65536 下并发砍半，损失远超量化误差。真实差多少要开 `--get-mismatch-metrics` 从 `tis_abs` 读，不是固定数字。

---

## Speculative Decoding

### Principle

decode 逐 token 生成、GPU 严重欠载。投机解码用 draft 批量猜多个 token、target 一次并行验证，接受率越高吞吐越大，且分布保持无偏（数学保证，见 QA）。这类模型（DeepSeek / Qwen 大 MoE）checkpoint 自带 MTP（Multi-Token Prediction）层，直接当 draft，接入 EAGLE 的 draft-verify 框架，不需要独立 draft 模型权重。draft 一轮：MTP 层自回归猜 N 步 → 拼成候选 token → target 一次 forward 并行验证 → 接受/纠正。

### Parameter Setting

**`speculative_algorithm = EAGLE`**：选 draft-verify 框架，draft 来源接内置 MTP 层。

**三个数字**：`speculative_num_steps = 4`（draft 自回归步数，每步采样一个 token）、`speculative_eagle_topk = 1`（每步保留候选数，1 = greedy 直链不分叉，>1 才展开成树）、`speculative_num_draft_tokens = 5`（送 target verify 的 token 数）。topk=1 时有硬约束（[eagle_worker_v2.py:265](python/sglang/srt/speculative/eagle_worker_v2.py#L265)）：

```python
assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1
```

`4 + 1 = 5`：4 步 draft 生成 4 个 token 加触发 token，verify 共 5 个。改 num_steps 必须同步改 num_draft_tokens，否则启动 assert 失败。

**`speculative_draft_attention_backend = nsa`**：draft（MTP 层）的 attention 独立配为 nsa（DSA），与 target 走同一套 DSA 路径、复用同一份 FP8 KV cache 和 flashmla 基础设施，不额外分配 draft-only KV 显存。

**`disable_overlap_schedule`**：投机解码必须关 overlap scheduler（[server_args.py:7043](python/sglang/srt/server_args.py#L7043) 的条件体现耦合）——draft-verify 是"宽请求"（同时占 verify slot 和 draft slot），打破了 overlap scheduler "每请求单步 forward" 的假设。colocate 和 PD decode 都跑投机解码所以都关；PD prefill 不做 decode-step 投机，未关。

### QA

**temperature 为什么不在这些参数里**。层次不同。`--speculative-*` 是引擎级、启动时固定，管 draft 怎么猜怎么验；temperature 是请求级采样参数，随每个请求传（slime 里走 `--rollout-temperature 1.0` 的 `sampling_params`），不写在引擎 flag 里。

**投机解码会不会影响 rollout 精度**。不会影响 token 分布，只带来微小 log-prob 数值噪声。分布保持是数学保证：temperature=0 贪心时接受的 token 与不投机完全一致，temperature>0 用修正的拒绝采样保证分布严格相等。唯一代价是 log-prob 的微小数值差异（verify 路径与 decode 路径浮点累加顺序不同），量级约 1e-3、叠在训推 mismatch 里被 TIS 覆盖。所以它换的是速度、不动分布。

**MTP 层在 RL 里训不训**。这套脚本不训。用 MTP 做 draft 不需要训练它就能用——MTP 预训练练好、随 checkpoint 加载，且分布保持使 draft 质量只影响接受率（速度）、不影响正确性。slime 的 `--enable-mtp-training` 默认 `False`（[arguments.py:1684](/Users/roy/Desktop/slime/slime/utils/arguments.py#L1684)），脚本没开：训练前向不传 mtp_labels（[model.py:539](/Users/roy/Desktop/slime/slime/backends/megatron_utils/model.py#L539)），MTP 拿不到梯度、冻结在加载值；但权重回传始终含 MTP 层（[common.py:188](/Users/roy/Desktop/slime/slime/backends/megatron_utils/update_weight/common.py#L188) 无条件处理 `mtp.layers.*`）。后果：主模型每步更新、策略漂移，MTP 冻结不变，rollout 正确性不受影响，但加速比随策略漂移衰减（冻结的 MTP 越来越猜不准）。想保持加速比就开：

```bash
--enable-mtp-training
--mtp-num-layers 1               # 必须同设，arguments.py:2216 的 assert
--mtp-loss-scaling-factor 0.2    # MTP loss 权重，默认 0.2
```

**衰减到极端会不会反而变慢，需不需要中途关掉**。会——接受率足够低时投机解码负加速。每轮固定开销是「draft num_steps 步 + 1 次 verify forward」，接受数趋近 0 时，这份开销比 plain decode 的单次 forward 还贵，存在盈亏平衡点。但不必然跌破：RL 微调 LR 小（1e-6）、权重漂移远小于预训练，MTP 又是共享主模型表征的强 draft，接受率会降但通常不崩。不需要预设"中途手动关"，而是监控加速比按需决策——`curl /server_info` 看 `avg_spec_accept_length` 的衰减曲线，三选一：开 `--enable-mtp-training` 让 MTP 跟训（从根上不衰减）；开 `--speculative-adaptive` 让引擎按 EMA 接受长度自动下调 num_steps（避免负加速）；或确实跌破平衡点又不想训 MTP，就关 `--speculative-algorithm` 回退 plain decode。
