# SGLang Feature Map

> 快照 commit `f2c875d1c8`（2026-07-15，`v0.5.15.post1` 之后的 main）。每项特性给出设计动机、核心机制与跳转入口（文档页 / 源码），按功能维度组织。

## Contents

- [Overview](#overview)

- [Inference Acceleration](#inference-acceleration)
  - [Speculative Decoding](#speculative-decoding)
  - [Adaptive Speculative Decoding](#adaptive-speculative-decoding)
  - [Attention Backend](#attention-backend)
  - [Weight Quantization](#weight-quantization)
  - [KV Cache Quantization](#kv-cache-quantization)
  - [CUDA Graph Optimization](#cuda-graph-optimization)
- [Parallelism](#parallelism)
  - [Expert Parallelism](#expert-parallelism)
  - [Pipeline Parallelism](#pipeline-parallelism)
  - [Multi-LoRA Serving](#multi-lora-serving)
  - [Multimodal DP](#multimodal-dp)
- [Architecture Disaggregation](#architecture-disaggregation)
  - [PD Disaggregation](#pd-disaggregation)
  - [EPD Disaggregation](#epd-disaggregation)
  - [HiCache](#hicache)
  - [HiSparse](#hisparse)
- [Startup Acceleration](#startup-acceleration)
  - [R-Fork](#r-fork)
  - [Checkpoint Engine](#checkpoint-engine)
- [Cluster Management](#cluster-management)
  - [SGLang Model Gateway](#sglang-model-gateway)
  - [llm-d Integration](#llm-d-integration)
- [Output Control](#output-control)
  - [Structured Outputs](#structured-outputs)
  - [Reasoning Parser](#reasoning-parser)
  - [Tool Parser](#tool-parser)
- [RL Integration](#rl-integration)
- [Debug and Ops](#debug-and-ops)
  - [Forward Hooks](#forward-hooks)
  - [Observability](#observability)
  - [Object Storage](#object-storage)

---

## Overview

| 类目 | 一句话 |
|------|--------|
| **Inference Acceleration** | 投机解码（EAGLE/MTP/NGRAM）、自适应投机解码、Attention 后端选择、权重量化、KV Cache 量化、CUDA Graph 优化——纵向堆叠覆盖 decode 每一层的加速手段。 |
| **Parallelism** | EP（all-to-all 路由 + DeepEP/Mooncake/NIXL 后端）、PP（异步 P2P + 动态 chunked prefill）、Multi-LoRA Serving（同 batch 混跑多适配器）、多模态 DP（ViT encoder 数据并行）。 |
| **Architecture Disaggregation** | PD 分离（prefill / decode 独立引擎 + RDMA KV 传输）、EPD 分离（再把 ViT encoder 独立出来）、HiCache（GPU→CPU→存储三级 KV 层次）、HiSparse（DSA 模型 CPU pinned 大 KV + GPU 热 buffer）。 |
| **Startup Acceleration** | R-Fork（运行中实例 GPU-to-GPU 直传权重，秒级扩容）、Checkpoint Engine（多节点并行加载 + 与 CUDA Graph capture overlap）。 |
| **Cluster Management** | SGLang Model Gateway（高性能路由网关，支持多协议/PD 路由/熔断限流/MCP/K8s）、llm-d（Kubernetes 层 prefix-aware 路由 + 分布式 KV 管理 + SLO 感知扩缩）。 |
| **Output Control** | 约束解码保证 JSON/Regex/EBNF 格式合法（XGrammar 默认）、Reasoning Parser 服务端剥离 `<think>` 思考链、Tool Parser 统一各模型 function call 输出格式。 |
| **RL Integration** | Engine sleep/wake 让出显存、三种权重写回策略（磁盘 / CUDA IPC / 分布式 broadcast）、partial rollout 暂停/恢复、确定性推理、cache-aware 负载均衡——支撑 slime/verl/OpenRLHF 的完整 rollout 侧 API。 |
| **Debug and Ops** | Forward Hooks（零改代码挂载中间激活导出）、Observability（Prometheus metrics / 请求日志 / crash dump 重放）、Object Storage（从 S3/GCS/Azure Blob 流式加载权重）。 |

---

## Inference Acceleration

### Speculative Decoding

推测解码（Speculative Decoding）解决的核心问题是：autoregressive decode 每步只生成一个 token，GPU 算力严重欠载。投机解码通过 draft 模型或 draft 机制批量预测多个候选 token，再由 target 模型一次性并行验证，接受率越高吞吐提升越大。

SGLang 支持四种算法：

| 算法 | 适用场景 | draft 来源 |
|------|----------|------------|
| EAGLE-2 | 通用，带独立 draft 模型 | 独立 EAGLE draft 模型 |
| EAGLE-3 | 更强草稿质量（推荐） | 独立 EAGLE3 draft 模型 |
| MTP | 模型自带 MTP 层（如 DeepSeek-V3/GLM-5.2） | 主模型内置 MTP head |
| DFLASH | 长序列 prefill 加速 | dual-chunk flash attention |
| NGRAM | 无需 draft 模型，输入重复模式场景 | 输入 token 的 n-gram 匹配 |

MTP（Multi-Token Prediction）与 EAGLE 的关键区别：MTP 不需要单独下载和部署 draft 模型，draft head 直接内嵌在主模型权重里，部署成本更低。以 GLM-5.2 为例，启用 `--speculative-algorithm EAGLE`、`--speculative-num-steps 4`、`--speculative-num-draft-tokens 5`，主模型自带的 MTP 层承担 draft 角色，无需额外参数。

EAGLE-3 相比 EAGLE-2 在高接受率场景下吞吐更高，是当前推荐选项（`--speculative-algorithm EAGLE3`）。

FR-Spec 变体（`--speculative-token-map`）通过频率排序缩小 draft 阶段的 lm_head 候选空间，减少 EAGLE-2 的 lm_head 计算开销。

参考文档：[docs_new/docs/advanced_features/speculative_decoding.mdx](docs_new/docs/advanced_features/speculative_decoding.mdx)
源码入口：[python/sglang/srt/speculative/](python/sglang/srt/speculative/)

---

### Adaptive Speculative Decoding

静态 `speculative_num_steps` 在负载变化时两头亏：batch size 小时步数过少、接受率高但轮次过早结束；batch size 大时步数过多、大量候选被拒绝、draft 计算白费。

自适应推测解码在运行时维护每个 batch size 档的 EMA（指数移动平均）接受率，自动切换 `num_steps` 档位，无需重启服务。

核心机制三层：

1. **AdaptiveSpeculativeParams**：EMA 策略，跟踪每个 batch size 档的平均接受长度。
2. **SpecRuntimeState**：每个步数档位的 CUDA graph 和 attention backend 绑定状态，切换时只做引用替换，不重新 capture。
3. **AdaptiveController**：在每次 verify 之后用 EMA 决策，通过 `warmup_batches` / `update_interval` / `down_hysteresis` 抑制过于频繁的档位切换。

当前限制：仅支持 `EAGLE` / `EAGLE3` 算法，且 `topk=1`。

配置示例：高接受率 draft 模型用激进配置，弱 draft 模型用保守默认配置。通过 `/server_info` 可实时查询当前档位与 EMA 接受率。

参考文档：[docs_new/docs/advanced_features/adaptive_speculative_decoding.mdx](docs_new/docs/advanced_features/adaptive_speculative_decoding.mdx)

---

### Attention Backend

不同注意力后端在 MHA / MLA、FP8 KV、投机解码、多模态等维度上的支持矩阵不同，没有一个后端在所有场景下都最优。SGLang 默认根据硬件和模型架构自动选最优后端，也支持 `--attention-backend` 手动指定。

**MHA 后端（标准多头注意力）：**

| 后端 | 亮点 | 限制 |
|------|------|------|
| FlashInfer | 通用、FP8 KV、投机 topk>1 | 不支持多模态 |
| FA3 (FlashAttention 3) | 多模态、FP8 KV、Hopper 优化 | — |
| FA4 (FlashAttention 4) | FP4 KV、Blackwell 原生 | page size 固定 128 |
| TRTLLM MHA | FP4 KV、原生 page size 16/32/64 | 不支持投机 topk>1 |
| Triton | 跨平台、FP8/FP4 KV | 无原生 page size |

**MLA 后端（DeepSeek 系列专用）：**

FlashMLA、TRTLLM MLA、FlashInfer MLA 分别针对 Hopper、Blackwell 等不同 GPU 优化。NSA（Native Sparse Attention，用于 DSA 模型如 GLM-5.2）有独立的 prefill backend（`flashmla_sparse`）和 decode backend（`flashmla_kv`），通过 `--attention-backend nsa` 启用。

参考文档：[docs_new/docs/advanced_features/attention_backend.mdx](docs_new/docs/advanced_features/attention_backend.mdx)

---

### Weight Quantization

量化分两类：**离线量化**（加载预量化权重）和**在线量化**（运行时动态计算缩放因子）。推荐使用离线量化，精度更可控，性能更高。

| 方法 | 类型 | 适用硬件 | 说明 |
|------|------|----------|------|
| FP8 | 离线/在线 | NVIDIA H100+、AMD MI300X+ | 最常用，精度损失小 |
| FP4 (MXFP4) | 离线 | NVIDIA Blackwell (B200) | 需要 `--quantization mxfp4` |
| AWQ | 离线 | 通用 | 激活感知权重量化 |
| GPTQ | 离线 | 通用 | 分组量化，兼容性广 |
| torchao | 在线 | 通用 | PyTorch 原生量化后端 |

使用预量化模型时不要同时加 `--quantization`。推荐从 Unsloth / NVIDIA ModelOpt / NeuralMagic 的 HuggingFace 合集获取经过验证的量化模型。

参考文档：[docs_new/docs/advanced_features/quantization.mdx](docs_new/docs/advanced_features/quantization.mdx)

---

### KV Cache Quantization

KV cache 是 decode 阶段显存占用最大的组成部分之一，尤其在长序列场景下。将 KV 从 BF16 量化为低精度格式，直接减少 KV 存储量，允许同时缓存更多 token，提升吞吐。

支持格式：

- **FP8 E4M3**：精度更高，动态范围较小（±240.0），推荐用于精度敏感场景。
- **FP8 E5M2**：动态范围更大（±57344.0），精度略低。
- **FP4 E2M1（MXFP4）**：4 bit 块状缩放格式，实验性，需要 Blackwell GPU。

内存节省数据（来自官方文档）：FP4 E2M1 相比 BF16 最高节省 3.56x 显存。

**关键约束**：dequantize 必须与 attention kernel fused，否则性能不升反降。使用前必须确认所选 attention backend 支持该量化格式（参见 Attention Backend 支持矩阵）。

启用方式：`--kv-cache-dtype fp8_e4m3`

参考文档：[docs_new/docs/advanced_features/quantized_kv_cache.mdx](docs_new/docs/advanced_features/quantized_kv_cache.mdx)

---

### CUDA Graph Optimization

CUDA Graph 消除 kernel launch 开销，在 decode 阶段（固定 batch size）收益显著。但 prefill/extend 的 token 数在每次迭代中都不同，标准 CUDA Graph 无法直接应用。

SGLang 提供三种变体：

**1. Piecewise CUDA Graph（分段图，当前默认）**

将模型前向拆成多个片段（大约每层一段，分割点通常是 MoE dispatch 算子），对每个预定义 token 数分别 capture 一张分段图。推理时将 input padding 到最近的预定义尺寸，逐段 replay。已默认开启，用 `--disable-piecewise-cuda-graph` 关闭。

**2. Breakable CUDA Graph（可中断图）**

允许在图执行中间插入条件分支（如 early exit、动态路由），解决标准图"全有全无"的限制。适用于需要运行时控制流的场景。

**3. Multimodal Encoder CUDA Graph**

专为 VLM 的 ViT encoder 捕获 CUDA Graph，减少视觉 encoder 的 kernel launch 开销。

参考文档：[docs_new/docs/advanced_features/piecewise_cuda_graph.mdx](docs_new/docs/advanced_features/piecewise_cuda_graph.mdx)、[docs_new/docs/advanced_features/breakable_cuda_graph.mdx](docs_new/docs/advanced_features/breakable_cuda_graph.mdx)、[docs_new/docs/advanced_features/cuda_graph_for_multi_modal_encoder.mdx](docs_new/docs/advanced_features/cuda_graph_for_multi_modal_encoder.mdx)

---

## Parallelism

### Expert Parallelism

MoE 模型（如 DeepSeek-V3、Qwen3-MoE）每层有大量 Expert，全部放在一张 GPU 上会超出显存。专家并行（EP）将 Expert 权重分布到多张 GPU，每个 GPU 只持有一部分 Expert，通过 all-to-all 通信把 token 路由到持有对应 Expert 的 GPU 上计算。

EP 的性能瓶颈在 all-to-all 通信效率。SGLang 支持以下 a2a 后端：

| 后端 | 适用场景 |
|------|----------|
| `deepep` | 大规模 EP，支持 normal（prefill 优化）和 low_latency（decode 优化，CUDA graph 兼容）两模式 |
| `mooncake` | 弹性 EP，基于 RDMA |
| `nixl` | 弹性 EP，RDMA + NVLink，支持故障容错 |
| `mori` | AMD ROCm 原生 |
| `flashinfer` | 大规模 EP |
| `none`（默认） | 混合 EP+TP（`ep_size < tp_size`）时使用 All-Reduce/All-Gather |

`--deepep-mode auto` 让调度器在 prefill 时自动选 normal 模式、decode 时自动选 low_latency 模式，是生产环境推荐配置。DeepEP、Mooncake、NIXL 当前要求 `ep_size = tp_size`。

**EPLB（Expert Load Balancing）**：MoE 路由天然不均匀，部分 Expert 过载而其他空闲。EPLB 在运行时重新分配 Expert 到 GPU，使每张卡的计算量趋于均衡，减少 EP 通信等待。

MoE 计算后端（`--moe-runner-backend`）：`auto`（默认，按硬件自选）、`triton`、`deep_gemm`（DeepGEMM，decode 侧性能更高，常在 decode override 里指定）。

参考文档：[docs_new/docs/advanced_features/expert_parallelism.mdx](docs_new/docs/advanced_features/expert_parallelism.mdx)
源码：[python/sglang/srt/layers/moe/](python/sglang/srt/layers/moe/)

---

### Pipeline Parallelism

超长序列（128K token 级别）的 prefill TTFT 极高，原因是前向计算复杂度随序列长度平方级增长，单节点 TP 无法完全消化。PP（流水线并行）将模型层切成多段，分布在不同节点上，不同 micro-batch 同时在各 stage 运行，降低 prefill 等待时间。

SGLang 的 PP 实现核心是**异步 P2P Micro-batching 事件循环**：

- 调度器使用非阻塞 `async_send` 发送 P2P 数据，返回 `P2PWork` handle，真正的 `wait()` 推迟到下次 batch 开始前才执行。
- 三条 CUDA stream 分工：`default_stream`（同步锚点）、`forward_stream`（GPU 前向）、`copy_stream`（D2H 拷贝），实现计算与通信 overlap。

**动态 Chunked Prefill**：固定 chunk size 在 PP 中会因 Attention 复杂度随前缀增长而产生越来越大的气泡。SGLang 用二次函数拟合运行时曲线，预测使每个 stage 执行时间对齐的下一个 chunk size，动态缩小后续 chunk，减少气泡传播。通过 `--enable-dynamic-chunking` 开启，平滑因子 `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`（默认 0.75，推荐范围 0.6-0.85）控制调整幅度。

实验数据（官方文档，NVIDIA H20，DeepSeek-V3.1）：128K 输入场景下，动态 chunking（初始 12K chunk、平滑因子 0.65）相比固定 4K chunk 进一步降低 TTFT；精确数字见 benchmark 报告。

参考文档：[docs_new/docs/advanced_features/pipeline_parallelism.mdx](docs_new/docs/advanced_features/pipeline_parallelism.mdx)

---

### Multi-LoRA Serving

单个 SGLang 实例服务多个 LoRA 适配器，同一 batch 中可混跑对应不同 LoRA 的请求，无需为每个适配器启动独立引擎。

实现基于 [S-LoRA](https://arxiv.org/pdf/2311.03285) + [Punica](https://arxiv.org/pdf/2310.18547)：LoRA 权重按适配器分组存储，批处理时通过 SGMV（Segmented Gather Matrix-Vector Multiply）内核一次性处理多个适配器的矩阵乘，避免逐适配器串行计算。

关键参数：

- `--max-loras-per-batch`：每 batch 最多同时使用的适配器数，影响 GPU 预留显存。
- `--enable-lora-overlap-loading`：异步 H2D 传输与 GPU 计算 overlap，适合适配器频繁切换场景。
- `--lora-eviction-policy`：LRU（默认）或 FIFO。
- `--lora-backend`：`triton` 或 `csgmv`（ChunkedSGMV）。

动态加载：服务运行期间可通过 API 加载新适配器，无需重启。

参考文档：[docs_new/docs/advanced_features/lora.mdx](docs_new/docs/advanced_features/lora.mdx)

---

### Multimodal DP

VLM（视觉语言模型）中 ViT encoder 的计算开销独立于语言模型。多模态 DP 对 VLM 的 encoder 部分启用数据并行：多个 DP worker 并行处理不同请求的图像 encoding，encoder 结果传给语言模型的 prefill 阶段。

通过 `--dp-size` 控制 encoder 的数据并行度。与 EPD 解聚合配合使用时，encoder 可独立水平扩缩。

参考文档：[docs_new/docs/advanced_features/dp_for_multi_modal_encoder.mdx](docs_new/docs/advanced_features/dp_for_multi_modal_encoder.mdx)

---

## Architecture Disaggregation

### PD Disaggregation

传统统一引擎将 prefill 和 decode 调度在同一批 GPU 上，带来两个结构性问题。第一，prefill batch 打断正在进行的 decode batch，decode 延迟抖动；第二，启用 DP-Attention 时，同一 DP 组里一个 worker 跑 prefill、另一个跑 decode，导致 decode 侧等待，TTFT 和 TBT 同时受损。

PD 解聚合将两个阶段部署在独立的 server group 上：prefill 实例只处理输入序列的 KV 计算，decode 实例接管 token 生成。prefill 完成后，KV cache 通过高速传输后端迁移到 decode 实例，后续 decode 不再有 prefill 干扰。

**KV 传输后端：**

- **Mooncake**：基于 RDMA/IB 的传输引擎，支持 NVLink（NVL72）、Intra-Node NVLink、RDMA 多种传输路径，通过环境变量 `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` 选择。
- **NIXL**：NVIDIA 官方传输框架，原生支持 RDMA 和 NVLink。

**典型部署形态（以 GLM-5.2 744B-A40B 为例）：**
- prefill：1 engine × 64 GPU，`deepep_mode=auto`，`load_balance_method=follow_bootstrap_room`
- decode：3 engine × 64 GPU，`deepep_mode=low_latency`，`moe_runner_backend=deep_gemm`，`load_balance_method=round_robin`

负载均衡策略差异：prefill 用 `follow_bootstrap_room`（decode 有空位才发 prefill），decode 用 `round_robin`，两者联动避免 prefill 积压。

参考文档：[docs_new/docs/advanced_features/pd_disaggregation.mdx](docs_new/docs/advanced_features/pd_disaggregation.mdx)
源码：[python/sglang/srt/disaggregation/](python/sglang/srt/disaggregation/)

---

### EPD Disaggregation

VLM 推理天然分三阶段：Encoder（ViT 图像编码，计算密集）、Prefill（语言模型 KV 初始化）、Decode（自回归生成，显存带宽密集）。现有 PD 解聚合把 Encoder 和 Prefill 绑定在一起，图像密集请求下 encoder 成为瓶颈，语言 prefill GPU 空等。

EPD（Encoder-Prefill-Decode）解聚合进一步把 ViT encoder 独立出来：encoder 实例单独水平扩缩，prefill 和 decode 各自独立部署，三层架构实现完全解耦。

encoder 输出通过 `--encoder-transfer-backend` 选择传输方式（默认 `zmq_to_scheduler`，Mooncake 版本支持 RDMA 直传）。

全局多模态 embedding cache（`--enable-mm-global-cache`）：encoder 实例把计算过的图像 embedding 存入 Mooncake，相同图像跨实例共享，无需重复 ViT 前向。

参考文档：[docs_new/docs/advanced_features/epd_disaggregation.mdx](docs_new/docs/advanced_features/epd_disaggregation.mdx)

---

### HiCache

长序列多轮对话场景下，KV cache 首先撑满 GPU HBM，其次溢出到 CPU，最终需要持久化到存储。三个层次的访问延迟差距是数量级的（HBM ns 级、CPU DRAM μs 级、NVMe ms 级），直接读写存储会拖慢 decode。

HiCache 构建三层 KV 缓存层次：GPU HBM → CPU pinned memory → 持久存储，通过预取和异步写回隐藏层间传输延迟。

存储后端：
- **Mooncake**：RDMA 加速，跨节点共享 CPU 池
- **3FS**（Fire-Flyer File System）：分布式高性能文件系统
- **NIXL**：NVIDIA 传输框架

适用场景：长上下文（100K+ token）多轮对话、共享系统 prompt 的大规模部署、KV cache 命中率高但 GPU 显存不足的场景。

参考文档：[docs_new/docs/advanced_features/hicache.mdx](docs_new/docs/advanced_features/hicache.mdx)、[docs_new/docs/advanced_features/hicache_design.mdx](docs_new/docs/advanced_features/hicache_design.mdx)

---

### HiSparse

长序列 decode 的显存压力来自每个请求持有完整 KV cache。并发数与单请求 KV 大小成反比：序列越长、并发越低。

HiSparse 针对使用 DeepSeek Sparse Attention（DSA）架构的模型（如 DeepSeek-V3.2、GLM-5.1、DeepSeek V4）：这类模型在 attention 时天然只选取部分 token 参与计算（稀疏 attention），因此只需在 GPU 上维护一个固定大小的"热 buffer"（如 4096 token），完整 KV 存放在 CPU pinned memory。

每次 decode 步骤：

1. 根据 attention score 选出 top-k 最相关的 token 位置。
2. CUDA kernel 从 CPU pinned memory 按需 swap-in 这些 KV 到 GPU buffer。
3. 在 GPU buffer 上完成 decode attention。
4. 异步把本步新生成的 KV 写回 CPU。

短序列（`seq_len ≤ device_buffer_size`）走快速路径，全部 KV 已在 buffer 中，无需 swap。

**与 PD 解聚合联动**：prefill 实例通过 RDMA 把 KV 直接写入 decode 实例的 CPU pinned memory pool，decode 侧 GPU 不参与 KV 接收，彻底消除 KV transfer 期间的 GPU 显存尖峰。

当前限制：需要 PD 解聚合模式，仅在 decode 实例上启用，模型需支持 DSA 架构。

参考文档：[docs_new/docs/advanced_features/hisparse_guide.mdx](docs_new/docs/advanced_features/hisparse_guide.mdx)

---

## Startup Acceleration

### R-Fork

大模型权重文件体积达数百 GB，从 NVMe/NFS 加载到 GPU 需要数分钟。每次扩容新实例都重走这条慢路径，制约弹性速度。

R-Fork（Tensor Remote Fork）将权重加载路径从"磁盘 → CPU → GPU"改为"运行中实例 GPU → 新实例 GPU"的直接 GPU-to-GPU 传输，零拷贝。源实例持续服务流量，新实例拉完权重后立即上线。

传输后端：

| 后端 | 说明 |
|------|------|
| `nccl` | 默认，基于 NCCL 建立通信组 |
| `transfer_engine` | Mooncake TransferEngine，基于 RDMA |
| `modelexpress` | 中央注册服务，无需手动配置 seed IP，自动发现和协调 P2P 传输 |

官方博客数据：DeepSeek-V3 等大模型启动从数分钟降到秒级（具体数字见 [R-Fork 博客](https://lmsys.org/blog/2025-12-10-rfork/)）。

使用方式：新实例启动时加 `--load-format remote_instance`，指定 seed 实例 IP 和端口。

参考文档：[docs_new/docs/advanced_features/rfork.mdx](docs_new/docs/advanced_features/rfork.mdx)

---

### Checkpoint Engine

大模型多节点部署时，每个节点各自从磁盘加载完整权重，磁盘 I/O 是瓶颈，且 CUDA graph capture 必须等权重加载完才能开始，串行等待浪费时间。

Checkpoint Engine 通过两个机制解决：

1. **多节点分片加载**：多个 torchrun 进程并行从磁盘读取不同分片，通过 broadcast 或 P2P 分发到推理进程。加入的节点越多，等效磁盘带宽越高。实测数据：DeepSeek-R1 在 H20-3e 双节点部署中加速 20 秒（来自官方文档）。

2. **与 CUDA graph capture overlap**：推理进程以 `--load-format dummy` 启动（先占位），同时开始 CUDA graph capture；checkpoint engine 在后台并行加载权重，两者 overlap 执行。

使用流程：推理进程加 `--wait-for-initial-weights` 和 `--load-format dummy` 启动；独立运行 checkpoint engine worker（通过 `torchrun` 或 `python -m sglang.srt.checkpoint_engine.update`）加载权重并推送。

参考文档：[docs_new/docs/advanced_features/checkpoint_engine.mdx](docs_new/docs/advanced_features/checkpoint_engine.mdx)

---

## Cluster Management

### SGLang Model Gateway

单实例 SGLang server 无法独立处理多实例负载均衡、故障切换、PD 解聚合路由等集群级问题。SGLang Model Gateway（前身为 Router）是专为 SGLang 优化的高性能路由网关，同时支持任意 OpenAI 兼容后端。

核心能力：

- **多协议数据面**：HTTP、gRPC、OpenAI 兼容，单网关同时代理三类流量。
- **PD 解聚合路由**：网关感知 prefill / decode worker 角色，自动把请求分发到正确的 worker 类型。
- **负载均衡策略**：cache-aware（RadixAttention 缓存命中率感知）、power-of-two、round-robin 等，可按模型/租户配置。
- **可靠性原语**：重试（指数退避 + jitter）、熔断器（per-worker）、令牌桶限流 + 排队。
- **MCP 集成**：原生 MCP client（STDIO/HTTP/SSE），多轮对话历史存储在网关侧，不泄漏给上游。
- **K8s 服务发现**：基于 Gateway API Inference Extension，通过 InferencePool 资源管理 worker 生命周期。
- **gRPC 全链路（Rust 实现）**：tokenizer、reasoning parser、tool parser 全部 in-process，延迟更低。
- **可观测性**：40+ Prometheus metrics、OpenTelemetry 分布式追踪、结构化日志。

多实例推理网关模式（`--enable-igw`）可同时管理多个模型的多个 router stack，实现多租户隔离。

参考文档：[docs_new/docs/advanced_features/sgl_model_gateway.mdx](docs_new/docs/advanced_features/sgl_model_gateway.mdx)

---

### llm-d Integration

单实例 RadixAttention 在单副本内最大化缓存复用，但多副本场景下 round-robin 负载均衡把相关请求分散到不同副本，缓存命中率崩塌。llm-d 是 CNCF Sandbox 项目（Red Hat、Google Cloud、IBM Research、CoreWeave、NVIDIA 联合发起），在 Kubernetes 层面为 SGLang 提供集群级优化。

llm-d 在 SGLang 之上增加的能力：

- **Prefix-aware 路由**：llm-d Router 评分每个副本的前缀缓存局部性，把请求路由到最可能已有该前缀缓存的副本，提升跨副本 RadixAttention 命中率。实测数据（官方基准，Llama 3.1 70B）：相比 round-robin，prefix-aware 路由提升 3x 输出吞吐、2x TTFT 加速。
- **分布式 KV cache 管理**：全局索引追踪每个 token block 在哪个副本，支持 KV tiered offload（CPU memory / 本地 SSD）。
- **PD 解聚合协调**：跨节点 KV 迁移通过高速互联完成，llm-d 负责调度。
- **SLO 感知自动扩缩**：基于推理队列深度和真实需求扩缩 SGLang 副本数，而不是 GPU 利用率。

使用 Gateway API Inference Extension 标准 Kubernetes 资源（Gateway、HTTPRoute、InferencePool）管理，兼容 Istio、GKE Inference Gateway 等标准网关。

参考文档：[docs_new/docs/advanced_features/llm-d.mdx](docs_new/docs/advanced_features/llm-d.mdx)

---

## Output Control

### Structured Outputs

模型生成的自由文本在下游需要 JSON 解析时频繁出错，程序无法稳定消费。结构化输出通过约束解码（constrained decoding）在 token 采样阶段实时 mask 掉违反约束的 token，保证输出格式合法。

支持三种约束格式：

- **JSON Schema**：Pydantic 模型或原生 JSON Schema，最常用。
- **Regex**：正则表达式约束任意输出模式。
- **EBNF**：扩展巴科斯范式，表达力最强，可描述复杂语法。

Grammar 后端：

| 后端 | 支持格式 | 性能 | 说明 |
|------|----------|------|------|
| XGrammar（默认） | JSON Schema、Regex、EBNF | 最高 | 推荐，GGML BNF 格式 |
| Outlines | JSON Schema、Regex | 中 | `--grammar-backend outlines` |
| Llguidance | JSON Schema、Regex、EBNF | 中 | `--grammar-backend llguidance` |

每次请求只能指定一种约束（`json_schema`、`regex`、`ebnf` 三选一）。建议在 prompt 中明确告知模型目标格式，以提升输出质量。

参考文档：[docs_new/docs/advanced_features/structured_outputs.mdx](docs_new/docs/advanced_features/structured_outputs.mdx)

---

### Reasoning Parser

DeepSeek R1、Qwen3、Kimi K2 等推理模型输出中混有 `<think>...</think>` 思考过程，下游应用通常只需要最终答案，手动解析容易出错。Reasoning Parser 在服务端自动剥离思考内容，通过 API 的 `reasoning_content` 字段单独返回，`content` 字段只包含最终回答。

已支持模型与对应 parser：

| 模型系列 | 思考标签 | Parser |
|----------|----------|--------|
| DeepSeek-R1 系列 | `<think>...</think>` | `deepseek-r1` |
| DeepSeek-V3 系列 | `<think>...</think>` | `deepseek-v3` |
| Qwen3 标准版 | `<think>...</think>` | `qwen3` |
| Qwen3-Thinking | `<think>...</think>` | `qwen3-thinking` |
| Kimi K2 Thinking | `◁think▷...◁/think▷` | `kimi_k2` |
| Apertus 2509 | `<\|inner_prefix\|>...<\|inner_suffix\|>` | `apertus2509` |

启用方式：`--reasoning-parser <parser_name>`。部分 parser 支持 `enable_thinking` 参数动态开关思维链。

参考文档：[docs_new/docs/advanced_features/separate_reasoning.mdx](docs_new/docs/advanced_features/separate_reasoning.mdx)

---

### Tool Parser

Function Calling（工具调用）格式在不同模型间存在差异：有的用 JSON、有的用 XML、有的有自定义 token。Tool Parser 在服务端统一解析各模型的 function call 输出，转换为标准 OpenAI tool call 格式返回。

通过 `--tool-call-parser <parser_name>` 指定，常用值：`llama3`、`qwen25`、`deepseek_v3`、`kimi_k2` 等。SGLang Model Gateway 的 gRPC 路径将 tool parser 内置在 Rust 进程内，延迟更低。

参考文档：[docs_new/docs/advanced_features/tool_parser.mdx](docs_new/docs/advanced_features/tool_parser.mdx)

---

## RL Integration

### SGLang for RL

RL 训练循环（rollout → 训练 → 权重同步）对推理引擎有特殊要求，常规 serving 引擎无法直接支持：训练时需要让出 GPU 显存给训练框架，权重更新后需要原地 refit 而不是重启，partial rollout 需要暂停和恢复生成。

SGLang 为 RL 场景提供以下 API：

**1. Engine Sleep / Wake**

启动时加 `--enable-memory-saver` 开启显存感知的 sleep/wake。`POST /release_memory_occupation` 释放显存（`tags` 取 `kv_cache` / `weights`，省略则全释放），让出 GPU 给训练；`POST /resume_memory_occupation` 按需唤回。底层通过 [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) 保留虚拟地址，避免 CUDA graph 重新 capture。调用释放前需保证引擎 idle（无 in-flight 请求）。

**2. Weight Refit**

三种权重写回策略，按部署形态选择，无需重启、无需重新 capture CUDA graph：

- `POST /update_weights_from_disk`：从磁盘 checkpoint 加载，最简单，适合弹性扩缩（新实例指向同一 checkpoint 即可）。
- `POST /update_weights_from_tensor`：colocate 训推共享 GPU 张量时用，CUDA IPC 零拷贝，最快。张量需经 `MultiprocessingSerializer.serialize` 序列化。
- `POST /update_weights_from_distributed`：分离部署，通过专用 NCCL/IB 通信组 broadcast。

**3. Partial Rollout（暂停/恢复生成）**

`POST /pause_generation` 停止接受新请求并等待 in-flight 完成，`POST /continue_generation` 放行。标准写回时序是 `pause_generation → update weights → continue_generation`，权重更新只能在引擎不处理推理时进行。

**4. 确定性推理**

`--enable-deterministic-inference` 降低跨 batch shape 的非确定性，缓解 runtime batching 和 kernel 选择带来的 token 概率漂移（training-inference mismatch 的来源之一）。要做到真正 on-policy，还需训练引擎使用相同的确定性 kernel。详见 [Deterministic Inference](docs_new/docs/advanced_features/deterministic_inference.mdx)。

**5. Cache-aware 负载均衡**

由 SGLang Model Gateway（router）提供，启动 router 时加 `--policy cache_aware`：优先路由到前缀缓存命中率最高的 worker，在 shared-system-prompt 场景下直接提升 rollout 吞吐。PD 解聚合下可用 `--prefill-policy` / `--decode-policy` 对两侧分别指定，可调阈值 `--cache-threshold`、`--balance-*`。已在 GLM 4.5+ 训练中生产验证。

这套 API 已在 slime（Kwai RL 训练框架）、verl、OpenRLHF 等多个 RL 框架中集成使用。

参考文档：[docs_new/docs/advanced_features/sglang_for_rl.mdx](docs_new/docs/advanced_features/sglang_for_rl.mdx)

---

## Debug and Ops

### Forward Hooks

调试大模型中间激活、验证量化精度损失、导出隐藏层到外部工具时，通常需要修改模型代码或手动插桩。Forward Hooks 通过 JSON 配置在不修改模型代码的情况下，给任意子模块挂载 PyTorch 标准 forward hook。

配置方式：在 `server_args` 的 `forward_hooks` 字段中指定目标模块路径（`target_modules`）和 hook 工厂函数（`hook_factory`，格式 `module.path:function_name`）。Hook 在 `ModelRunner.initialize` 时注册，每次 forward pass 都触发。

```json
{
  "forward_hooks": [
    {
      "name": "capture-layer-0-output",
      "target_modules": ["model.layers.0"],
      "hook_factory": "my_hooks:export_activation",
      "config": {"output_path": "/tmp/layer0.pt"}
    }
  ]
}
```

参考文档：[docs_new/docs/advanced_features/forward_hooks.mdx](docs_new/docs/advanced_features/forward_hooks.mdx)
源码：[python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py)

---

### Observability

**Prometheus Metrics**

启用 `--enable-metrics` 后，`/metrics` 端点暴露服务端指标，涵盖请求吞吐（TPS）、token 吞吐（TTPS）、队列深度、KV cache 使用率、prefill/decode 延迟分位数等。SGLang Model Gateway 额外提供 40+ 路由层指标。

**请求日志**

默认不记录请求内容。`--log-requests` 开启请求级日志，`--log-request-level` 控制详细程度，运行时通过 `python -m sglang.srt.managers.configure_logging` 动态调整，无需重启。

**Crash Dump & Replay**

`--crash-dump-folder` 开启 crash dump：服务崩溃时自动保存崩溃前 5 分钟的所有请求（pickle 格式）。通过 `scripts/playground/replay_request_dump.py` 重放，复现并调试崩溃原因。`--dump-requests-folder` + `--dump-requests-threshold` 支持常态化请求 dump，用于线下压测回放。

参考文档：[docs_new/docs/advanced_features/observability.mdx](docs_new/docs/advanced_features/observability.mdx)

---

### Object Storage

大模型权重文件从本地磁盘加载要求预先下载完整模型，占用大量本地存储，不适合弹性扩缩场景。Object Storage 支持直接从 S3、GCS、Azure Blob 等对象存储拉取权重，跳过本地落盘。

加载机制分两阶段：

1. **元数据下载**（一次性，进程启动前）：config、tokenizer 等小文件下载到本地缓存。
2. **权重流式加载**（lazy，模型初始化期间）：通过 `runai_streamer` 按需从对象存储拉取权重张量，不全量下载。

支持 URI 格式：`s3://`、`gs://`、`az://`。提供对象存储 URI 时，`--load-format runai_streamer` 会自动检测，通常可省略。

```bash
python -m sglang.launch_server \
  --model-path s3://my-bucket/models/deepseek-v3/ \
  --tp-size 8
```

参考文档：[docs_new/docs/advanced_features/object_storage.mdx](docs_new/docs/advanced_features/object_storage.mdx)
