## Contents

- [Log-Probs OOM: Anatomy of the fused CE Peak](#log-probs-oom-anatomy-of-the-fused-ce-peak)
  - [The Key Parameter: `--log-probs-chunk-size`](#the-key-parameter---log-probs-chunk-size)
  - [Memory Estimates](#memory-estimates)
  - [Mitigations](#mitigations)
- [Debug Flags: Decouple Rollout and Train](#debug-flags-decouple-rollout-and-train)
- [GDN Layer Context Parallel: The All-Gather Trap](#gdn-layer-context-parallel-the-all-gather-trap)
  - [Current Behavior: All-Gather Before Compute](#current-behavior-all-gather-before-compute)
  - [Memory Analysis: CP Saves Nothing Here](#memory-analysis-cp-saves-nothing-here)
  - [Sequential Ring State Passing (Recommended Fix)](#sequential-ring-state-passing-recommended-fix)
- [Weight Lifecycle: raw, bridge, and the Actor Init Fallback Chain](#weight-lifecycle-raw-bridge-and-the-actor-init-fallback-chain)
  - [Startup Weight Flow](#startup-weight-flow)
  - [raw vs bridge](#raw-vs-bridge)
  - [Actor Init Fallback Chain](#actor-init-fallback-chain)
- [Normalization: Three Knobs at Three Stages](#normalization-three-knobs-at-three-stages)
- [remove_sample: Reject Sampling Without Shrinking the Batch](#remove_sample-reject-sampling-without-shrinking-the-batch)
  - [Mechanism](#mechanism)
  - [The `zeros_like` + backward Bug](#the-zeros_like--backward-bug)
- [Train-Infer Mismatch: TIS, IcePop, and KPop](#train-infer-mismatch-tis-icepop-and-kpop)
  - [Three Algorithms](#three-algorithms)
  - [TIS Entry Point](#tis-entry-point)
  - [IcePop vs KPop](#icepop-vs-kpop)
  - [CLI 陷阱](#cli-陷阱)
- [REINFORCE++ vs GRPO: Algorithm Differences and Switchover](#reinforce-vs-grpo-algorithm-differences-and-switchover)
  - [The Three Divergence Points](#the-three-divergence-points)
  - [Degeneration: REINFORCE++ at kl=0 == GRPO](#degeneration-reinforce-at-kl0--grpo)
  - [Switchover Checklist](#switchover-checklist)
- [Training Issues Compendium](#training-issues-compendium)
  - [Parallel Configuration](#parallel-configuration)
  - [Memory and Resource](#memory-and-resource)
  - [Ray Cluster and Infrastructure](#ray-cluster-and-infrastructure)
  - [Checkpoint and Model Loading](#checkpoint-and-model-loading)
  - [Framework Regression Bugs](#framework-regression-bugs)
- [Observability: wandb Metrics Reference](#observability-wandb-metrics-reference)
  - [rollout/ 前缀](#rollout-前缀rollout-阶段记录)
  - [train/ 前缀](#train-前缀训练阶段记录)
  - [Response Length: Two Different Counts](#response-length-two-different-counts)

---

## Log-Probs OOM: Anatomy of the fused CE Peak

长序列 + 大词表场景下，per-token log-probs 计算是最常见的 OOM 触发点。fused cross-entropy 内部在 fp32 中间 buffer 上的峰值显存远超模型权重本身的占用。

### The Key Parameter: `--log-probs-chunk-size`

- 定义：[slime/utils/arguments.py:157](slime/utils/arguments.py#L157)，`type=int, default=-1`
- 消费：[slime/backends/megatron_utils/loss.py:425](slime/backends/megatron_utils/loss.py#L425) `chunk_size = args.log_probs_chunk_size`

默认 `-1` 表示不切块，整条序列一次性走 fused CE。长序列 + 大词表场景务必显式设一个正值（例如 `1024` / `2048`）。

### Memory Estimates

定义：`S` = 该 micro-batch 进入 fused CE 的 token 数，`V_tp = vocab_size / TP_size`（每个 TP rank 持有的词表分片）。`micro-batch-size=1` 不开 CP 时 `S = seq_len`；开 CP 时 `S = seq_len / cp_size`。

单块 fp32 中间 buffer：

```
mem_fp32 = S · V_tp · 4 B
```

no-chunk 分支峰值（同时叠着原始 logits + `logits.clone()` + fused CE 内部 fp32）：

```
peak ≈ S · V_tp · 8 B    # with_entropy=False
peak ≈ S · V_tp · 14 B   # with_entropy=True（额外 clone + entropy 内部 fp32）
```

以 `V_tp = 124160`（Qwen 系列 TP 切完后的典型词表分片）为例：

| seq_len (S) | 仅 fp32 buffer | no-chunk 峰值（无熵） | no-chunk 峰值（含熵） |
|---|---|---|---|
| 15,000 | ~6.94 GiB | ~13.9 GiB | ~24.3 GiB |
| 20,000 | ~9.25 GiB | ~18.5 GiB | ~32.4 GiB |
| 25,000 | ~11.56 GiB | ~23.1 GiB | ~40.5 GiB |
| 30,000 | ~13.88 GiB | ~27.8 GiB | ~48.6 GiB |

一张 80 GiB 的卡还要同时承载模型权重、优化器状态分片、activation、grad buffer，长序列下光这一步就能顶穿。开 chunk 后峰值降到 `chunk_size · V_tp · 4 B`，例如 `chunk_size=2048, V_tp=124160` → ~0.95 GiB，基本可控。

### Mitigations

- 显式设置 `--log-probs-chunk-size`（例如 `1024` / `2048`），走 chunk 分支。
- 仅在必要时开熵（`with_entropy`），熵路径还有一次 [`logits.clone()` — ppo_utils.py:672](slime/utils/ppo_utils.py#L672)。
- 排查同卡残留进程（例如 sglang offload 未干净），`nvidia-smi` 确认实际可用显存。
- 降 `--max-tokens-per-gpu` / micro-batch，或设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片。

## Debug Flags: Decouple Rollout and Train

slime 把 RL pipeline 拆成「rollout / train」两段可独立跑的管道，靠以下 flag 组合（全部定义在 `add_debug_arguments` — [slime/utils/arguments.py:1100](slime/utils/arguments.py#L1100)）：

| Flag | 作用 | 适用场景 |
|------|------|----------|
| `--debug-rollout-only` | 只跑 rollout（不起训练 actor），见 [arguments.py:1401](slime/utils/arguments.py#L1401)、[rollout.py:488](slime/ray/rollout.py#L488) | 验证数据生成、奖励函数、过滤器；不需要 Megatron |
| `--debug-train-only` | 只跑训练（不起 SGLang），见 [arguments.py:1402](slime/utils/arguments.py#L1402)、[placement_group.py:83](slime/ray/placement_group.py#L83) | 复现训练侧 loss/反向/优化器 bug |
| `--save-debug-rollout-data` / `--save-debug-train-data` | 把 `Sample` 列表和 `rollout_data` dump 成 `.pt`（[train_dump_utils.py:9](slime/utils/train_dump_utils.py#L9)、[rollout.py:637](slime/ray/rollout.py#L637)） | 把线上一轮数据落盘，供离线回放 |
| `--load-debug-rollout-data` | 直接从 `.pt` 读回 rollout，跳过 SGLang 整个链路（[rollout.py:569](slime/ray/rollout.py#L569)，[arguments.py:1618](slime/utils/arguments.py#L1618) 会自动置 `debug_train_only=True`） | **最重要**：离线复现训练问题，秒级启动 |
| `--dump-details` | 一键同时开启上面两个 save（[arguments.py:1614-1616](slime/utils/arguments.py#L1614-L1616)） | 一次性保全现场 |

推荐的两步调试范式：

```bash
# 步骤 1：线上跑一次，把 rollout 数据 dump 下来
--dump-details /tmp/slime_dump

# 步骤 2：后续所有训练侧改动，直接回放，不再起 sglang
--load-debug-rollout-data /tmp/slime_dump/rollout_data/{rollout_id}.pt \
--load-debug-rollout-data-subsample 0.1   # 只用 10% 数据，迭代更快
```

Dump 落盘点参考：

- rollout 侧：[`RolloutManager._save_debug_rollout_data` — slime/ray/rollout.py:637](slime/ray/rollout.py#L637)
- 训练侧：[`save_debug_train_data` — slime/utils/train_dump_utils.py:9](slime/utils/train_dump_utils.py#L9)（被 [actor.py:494](slime/backends/megatron_utils/actor.py#L494) 调用）

---

## GDN Layer Context Parallel: The All-Gather Trap

GDN（Gated Delta Network）层在开 CP 时的显存行为与 attention 层完全相反：attention 的 ring-pass 让每个 rank 只处理自己那段序列，显存随 cp_size 线性降低；GDN 层的现行实现则让每个 rank 都计算完整序列，显存不降反升。

### Current Behavior: All-Gather Before Compute

核心逻辑在 [hf_attention.py:117-143](slime/backends/megatron_utils/gdn/hf_attention.py#L117-L143)。当 `cp_size > 1` 时，流程如下：

```python
# forward() 在 CP > 1 时
hidden_states_list = _AllGatherForDuplicatedComputation.apply(
    hidden_states, mpu.get_context_parallel_group()
)
# 重新拼出 whole_hidden_states（完整序列）
hidden_states = torch.cat(whole_hidden_states_list, dim=0)  # [total_seq, hidden]

# 整段送进 GDN kernel
output = self.hf_forward(hidden_states, packed_seq_params)
# → chunk_gated_delta_rule(q, k, v, g, beta, initial_state=None, ...)

# forward 结束后按 zigzag 切回本 rank 的 chunk
chunks = torch.chunk(seq, 2 * cp_size, dim=0)
output_list.append(chunks[cp_rank])
output_list.append(chunks[2 * cp_size - 1 - cp_rank])
```

这不是 all-reduce，是 **all-gather → full-sequence-compute → local-slice-back**。每个 rank 在 GPU 上持有并独立计算了全量序列。backward 时 `_AllGatherForDuplicatedComputation` 的自定义反传返回 `grads[ctx.rank]` 而非 reduce-scatter，目的是避免梯度被 ×cp_size 放大。

### Memory Analysis: CP Saves Nothing Here

| 节点 | 正常 CP（attention） | GDN 现行做法 |
|---|---|---|
| hidden_states 进层 | `total_seq / cp_size` | `total_seq`（all-gather 后） |
| GDN kernel 中间激活（q/k/v/chunk state） | — | `total_seq` 级别 |
| backward 激活存储 | `1/cp_size` | full sequence |
| 计算量 | `1/cp_size`（attention ring-pass） | full（每 rank 各跑一遍） |

peak 显存在 GDN 层是正常预期的 `cp_size` 倍，且计算没有被并行摊薄。数字举例：`cp_size=8`，GDN 层显存就是单卡无 CP 时的 8 倍。

根因在 `initial_state=None + output_final_state=False`：recurrent 状态根本没有跨 rank 传递，每个 rank 把序列从头算起。对 packed seq 场景（序列间独立）语义上是正确的，但完全抵消了 CP 应带来的显存收益。

### Sequential Ring State Passing (Recommended Fix)

把"all-gather 全序列、每 rank 重复算"改成"每 rank 算自己的 chunk，通过 P2P 传递 recurrent state"。

```
rank 0: chunk_0 → final_state_0 ──P2P send──► rank 1
rank 1: initial_state=final_state_0, chunk_1 → final_state_1 ──► rank 2
...
rank n-1: initial_state=final_state_{n-2}, chunk_{n-1} → done
```

**改动 1：数据布局从 zigzag 改为 contiguous。**

zigzag 是为 causal attention 负载均衡设计的——每个 rank 持有序列头尾两段来平衡前后 token 的 attention 计算量。recurrent 模型没有这个需求，必须按序列顺序处理：rank r 持有 `[r * L/cp, (r+1) * L/cp]` 的连续 chunk。

**改动 2：在 `HuggingfaceAttention.forward` 里替换 all-gather 为 ring P2P。**

```python
if mpu.get_context_parallel_world_size() > 1:
    cp_rank  = mpu.get_context_parallel_rank()
    cp_size  = mpu.get_context_parallel_world_size()
    cp_group = mpu.get_context_parallel_group()

    # 1. recv 上一 rank 的 final_state（rank 0 用 None）
    initial_state = None
    if cp_rank > 0:
        initial_state = recv_state_from(cp_rank - 1, group=cp_group)

    # 2. 在本地 chunk 上跑 GDN，开启 output_final_state
    output, final_state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=local_cu_seqlens,
    )

    # 3. 把 final_state send 给 rank+1（非最后一 rank）
    if cp_rank < cp_size - 1:
        send_state_to(final_state, cp_rank + 1, group=cp_group)
```

**通信量对比**：现行 all-gather 每层传 `total_seq × hidden_size` 的数据量；ring P2P 每层只传 `[num_heads, head_k_dim, head_v_dim]` 的 state 张量。以 Pro 为例，state 约为 `num_heads × 64 × 64 × 2B ≈ 几 MB`，与全序列 all-gather 相比量级悬殊。

**代价**：rank 间产生串行依赖——rank k 必须等 rank k-1 的 final_state 到位才能开始计算。这不是实现缺陷，而是 recurrent 模型的本质约束。all-gather 做法只是用 `cp_size` 倍显存和 `cp_size` 倍计算换了表面上的"并行"——每个 rank 仍在完整重跑全序列，不存在真正的并行加速。ring 方案把计算量降到 `1/cp_size`，代价是引入 `cp_size - 1` 次 P2P 通信的串行延迟。

---

## Weight Lifecycle: raw, bridge, and the Actor Init Fallback Chain

slime 同时持有两份权重：HF 格式（给 SGLang 和 tokenizer）与 Megatron 分片格式（给训练 actor）。两份权重的加载和同步逻辑相互独立，搞混了会在冷启时踩坑。

### Startup Weight Flow

启动分三步，与 `--colocate` 无关：

1. **SGLang engine 用 `--hf-checkpoint` 直接加载 HF safetensors**（[sglang_engine.py:520](slime/backends/sglang_utils/sglang_engine.py#L520)），`raw` / `bridge` 两种 mode 都走这条路。
2. **Megatron actor 按 `--load` / `--ref-load` 的格式加载**（[checkpoint.py:106-130](slime/backends/megatron_utils/checkpoint.py#L106-L130)），靠 `_is_megatron_checkpoint()` 判断是 Megatron dist-ckpt 还是 HF 目录。
3. **训练前无条件推一次权重给 SGLang**（[train.py:30](train.py#L30)），覆盖掉步骤 1 加载的 HF 权重。

`--hf-checkpoint` 真正的持久角色是架构 schema + tokenizer，而非权重本身。第一次 rollout 之前 SGLang 用的已经是 Megatron 刚加载的权重。

### raw vs bridge

`--megatron-to-hf-mode` 决定 Megatron ↔ HF 张量转换方式，对应两条独立代码路径：

| | `raw` | `bridge` |
|---|---|---|
| 转换实现 | 手工 `all_gather` + `convert_to_hf` 改名 | `AutoBridge` 自动处理 gather/改名/reshape |
| 冷启加载格式 | **只认 torch_dist**，HF 目录直接 assert 失败 | torch_dist 和 HF 目录都认 |
| 离线转换工具 | **必须先跑** `convert_hf_to_torch_dist.py` | 不需要 |
| 分离部署权重同步 | `UpdateWeightFromDistributed`，NCCL broadcast | 硬编码 `convert_to_hf`，**只支持 `raw`** |

`bridge` 模式在分离部署下（`colocate=False`）目前无法使用——`UpdateWeightFromDistributed` 调用了硬编码的 `convert_to_hf`（[update_weight_from_distributed.py:162](slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L162)），bridge 模式必须配 colocate 使用。

### Actor Init Fallback Chain

真正被加载的参数是 `args.load`，回退链定义在 [arguments.py:1643-1668](slime/utils/arguments.py#L1643-L1668)：

- **raw 模式**：`--load`（必须 torch_dist）→ `--ref-load`（torch_dist，强制 `finetune=True` / `no_load_optim=True` / `no_load_rng=True`）。`--hf-checkpoint` 不参与 actor 权重加载。
- **bridge 模式**：`--load`（torch_dist 或 HF 目录）→ `--ref-load` → `--hf-checkpoint`，链上任意一级都可以是两种格式之一。

ref model（KL 用）走独立路径：[actor.py:107-108](slime/backends/megatron_utils/actor.py#L107-L108) 的 `load_other_checkpoint("ref", args.ref_load)`，仅在 `kl_coef != 0 or use_kl_loss` 时触发，没有回退。"actor 初始权重"和"ref model 权重"是两件独立的事，`--ref-load` 可能同时承担、只承担其一，或不需要（GRPO 关 KL 时）。

| 组合 | actor init | ref model（KL≠0 时） | 需要离线转换 |
|---|---|---|---|
| `raw` + 冷启 | `--ref-load` torch_dist | 同左 | **必须** |
| `raw` + 续训 | `--load` torch_dist | `--ref-load` torch_dist | 必须 |
| `bridge` + 冷启 + 无 KL | `--load` 直接指 HF | 不需要 | **不需要** |
| `bridge` + 冷启 + 有 KL | `--load` HF 或 torch_dist | `--ref-load` HF 或 torch_dist | 不需要 |
| `bridge` + 续训 | `--load` torch_dist | `--ref-load` HF 或 torch_dist | 视是否有 ref |

---

## Normalization: Three Knobs at Three Stages

slime 里"归一化"这个词指三件完全不同的事，作用在不同阶段，用错了要么梯度消失要么梯度爆炸。

| 参数 | 阶段 | 作用 | 生效的 estimator |
|---|---|---|---|
| `--disable-rewards-normalization`（`dest=rewards_normalization`，默认 True） | rollout，[rollout.py:655-680](slime/ray/rollout.py#L655-L680) | 同一 prompt 的多个 response reward 做组内减均值 | `grpo` / `gspo` / `reinforce_plus_plus_baseline` |
| `--normalize-advantages`（默认 False） | training，[loss.py](slime/backends/megatron_utils/loss.py) `compute_advantages_and_returns()` | 全局所有 GPU 所有 token 做零均值 + 单位方差（all_reduce 聚合统计量） | 任意 estimator |
| `--disable-grpo-std-normalization`（`dest=grpo_std_normalization`，默认 True） | rollout，同上 | 组内减均值后再除以 std（Dr.GRPO） | 仅 `grpo` / `gspo` |

**REINFORCE++ 必须开 `--normalize-advantages`**：token-level 折扣累积回报数值范围可达几十到上百，不开会导致梯度爆炸。

**RFT 场景必须关 `--normalize-advantages`**：如果 filter 之后只剩 reward=1 的正样本，advantage 全是同一个值，全局 whiten 后全变零 → 没有梯度。

| 场景 | estimator | rewards_norm | advantage_norm | grpo_std_norm |
|---|---|---|---|---|
| GRPO 默认 | `grpo` | ✅ 组内减均值 | 可选 | ✅ 除以 std |
| PPO | `ppo` | ❌ | ❌（GAE 自带） | ❌ |
| REINFORCE++ | `reinforce_plus_plus` | ❌ | ✅ **必须开** | ❌ |
| RFT（正样本 SFT） | `reinforce_plus_plus` | ❌ | ❌ **必须关** | ❌ |

---

## remove_sample: Reject Sampling Without Shrinking the Batch

`remove_sample=True` 不从 batch 中物理移除样本，而是把该样本的 `loss_mask` 全部置零。batch size 不变，这条样本的损失贡献为零。

### Mechanism

调用链（按顺序）：

1. rollout filter（`--rollout-sample-filter-path`）把 `sample.remove_sample = True`，[sglang_rollout.py:471-473](slime/rollout/sglang_rollout.py#L471-L473)
2. `_convert_samples_to_train_data` 遍历 samples，`remove_sample=True` 时 `loss_mask = [0] * response_length`，[rollout.py:716-718](slime/ray/rollout.py#L716-L718)
3. reward normalization（步骤 1 之后）发生时，被标记的 sample 的 reward **仍然参与了组内均值计算**——这是一个需要注意的语义点
4. `get_reinforce_plus_plus_returns` 遇到全零 mask 需要特殊处理（见下方 bug 节）

内置实现：[rollout/filter_hub/reject_sampling_filter.py](slime/rollout/filter_hub/reject_sampling_filter.py) 的 `filter_low_reward()`，把 reward < 0.9 的 sample 标记 `remove_sample=True`。

### The `zeros_like` + backward Bug

开启 `remove_sample` + REINFORCE++ 会触发两阶段 bug：

**第一阶段**：`get_reinforce_plus_plus_returns` 用 `full_mask.nonzero()` 找最后一个有效 token 位置，全零 mask 找不到 → `AssertionError: Sequence at index 0 is fully masked`。

**第二阶段**（初次用 `zeros_like` 修复后）：`torch.zeros_like(local_kl_chunk)` 会继承源 tensor 的 autograd graph，Megatron pipeline 反向传播遍历到已释放的子图 → `RuntimeError: Trying to backward through the graph a second time`。

正确修法：返回 **detach 后**的零张量，且形状匹配 CP 场景的 local chunk：

```python
if full_mask.sum().item() == 0:
    local_returns_chunk = torch.zeros_like(local_kl_chunk).detach()
    final_returns_chunks.append(local_returns_chunk)
    continue
```

`.detach()` 切断 autograd graph 关联，`zeros_like(local_kl_chunk)` 自动匹配 CP local chunk 形状，避免后续 `torch.cat` 时 shape mismatch。

---

## Train-Infer Mismatch: TIS, IcePop, and KPop

SGLang 生成时的 logprobs（`π_infer`）和 Megatron 训练时重算的 logprobs（`π_train`）因量化精度、attention 实现差异而不完全一致。用 `π_train` 做 old policy 时，重要性采样比率引入噪声，长时间训练会导致策略崩溃。

### Three Algorithms

| | `use_rollout_logprobs` | `use_rollout_correction` | 描述 | old_log_probs 来源 |
|---|---|---|---|---|
| 标准 PPO | False | False | 训练引擎重算 old logprobs | Megatron |
| Bypassing PPO | True | False | 直接用 rollout logprobs 作 old policy，跳过重算 | SGLang |
| Decoupled PPO | False | True | 三角色：target π_θ、proximal π_old（重算）、behavior π_SGLang | Megatron |

**`--use-rollout-logprobs` 与 `--use-tis` 互斥**（[arguments.py:1590](slime/utils/arguments.py#L1590)）——Bypassing PPO 和 Decoupled PPO 不能叠加。

### TIS Entry Point

所有修正函数都挂在 `policy_loss_function` 的同一个接管点（[loss.py:968-996](slime/backends/megatron_utils/loss.py#L968-L996)）。三个函数签名和返回值完全一致，可无缝替换：

- 默认：[`vanilla_tis_function`](slime/backends/megatron_utils/loss.py#L753)——token-level truncate，`torch.clamp(tis, min=tis_clip_low, max=tis_clip)`，区间外仍贡献梯度（权重被压到边界）
- 可选：[`icepop_function`](slime/backends/megatron_utils/loss.py#L777)——超出 `[tis_clip_low, tis_clip]` 的 token 权重置 0（`torch.where`），不贡献梯度
- 可选：[`kpop_function`](slime/backends/megatron_utils/loss.py#L803)——用对称二元 KL 替换固定比率约束，自适应容差

`icepop_function` 和 `kpop_function` 都没有 CLI flag，必须通过 `--custom-tis-function-path` 指过去才能用。

### IcePop vs KPop

IcePop 的问题：固定比率阈值 `[tis_clip_low, tis_clip]` 对所有 token 一视同仁，但低概率 token 的固有噪声更大，导致 IcePop 过度掩掉低概率 token，且随训练推进 train-infer gap 扩大时掩码率反而下降。

KPop 用对称二元 KL 解决这个问题。把词表视为「这个 token」vs「其余全部」的二事件分布：

$$D_{KL}^B(p \| q) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$$

掩码条件要求正反两方向都小于阈值 φ，只有一个超参 `--kpop-phi`（默认 0.1 nats）。高概率 token 容差收紧，低概率 token 容差自然放宽。

切换方式：

```bash
# IcePop
--use-rollout-logprobs --get-mismatch-metrics \
--custom-tis-function-path slime/backends/megatron_utils/loss.py:icepop_function \
--tis-clip-low 0.5 --tis-clip 2.0

# KPop
--use-rollout-logprobs --get-mismatch-metrics \
--custom-tis-function-path slime/backends/megatron_utils/loss.py:kpop_function \
--kpop-phi 0.1
```

### CLI 陷阱

文档里写的很多参数实际**没有注册到 argparse**，直接传会报 `unrecognized arguments`：

- `--use-rollout-correction`：不存在，只是 README 概念占位符
- `--tis-lower-bound`、`--tis-upper-bound`、`--tis-mode`、`--tis-level`、`--tis-batch-normalize`：这些是 `mis.yaml` 的 YAML 键，必须配合 `--custom-config-path` 注入
- 整套 `--rs-*` flag：完全在 `examples/train_infer_mismatch_helper/mis.py` 里，主仓 CLI 没有
- `--get-mismatch-metrics` 强制依赖 `--custom-tis-function-path`（[arguments.py:1593-1596](slime/utils/arguments.py#L1593-L1596)），只设前者不设后者直接 assert 失败

---

## REINFORCE++ vs GRPO: Algorithm Differences and Switchover

两种 estimator 在代码层面有三个分叉点，其余（policy loss、checkpoint、rollout 逻辑）完全共享。

### The Three Divergence Points

**分叉 1：rollout 阶段 reward 后处理**（[rollout.py:655-680](slime/ray/rollout.py#L655-L680)）

GRPO / reinforce_plus_plus_baseline 做组内减均值，GRPO 还可选除以 std（Dr.GRPO）。REINFORCE++ 不走这个分支，reward 原样传下去。

**分叉 2：advantage 计算**（[loss.py:571-741](slime/backends/megatron_utils/loss.py#L571-L741)）

| | GRPO | REINFORCE++ |
|---|---|---|
| advantage 来源 | 标量 reward 广播到所有 token：`torch.ones_like(kl) * reward` | 每个 token 有独立的 discounted return：逆序 `G_t = r_t + γ·G_{t+1}` |
| KL 惩罚位置 | loss 层面（`--kl-loss-coef`） | return 内部（`--kl-coef`），每个 token 有 `-kl_coef * KL_t` |
| 最终 token | 无特殊处理 | 额外加上序列级标量 reward |

**分叉 3：advantage whitening**

REINFORCE++ 的 token-level return 数值范围可达几十到上百，**必须开 `--normalize-advantages`**。GRPO 组内 norm 后可不再 whiten。

### Degeneration: REINFORCE++ at kl=0 == GRPO

当 `kl_coef=0` 时，所有中间 token 的 reward 为 0，只有最后一个 token 有标量 reward。逆序累加后每个 token 的 return 都等于同一个标量——与 GRPO 的 `torch.ones_like(kl) * reward` 完全等价。再加 group normalization 就得到 GRPO。

```
REINFORCE++ (kl>0, gamma<1)   → token-level discounted return + KL penalty
    ↓ kl→0
REINFORCE++ (kl=0)             → 所有 token 共享标量 reward
    ↓ + group normalization
GRPO (无 std norm)
    ↓ + std normalization
GRPO (Dr.GRPO)
```

### Switchover Checklist

从 GRPO 切到 REINFORCE++：

```bash
# 修改点
--advantage-estimator reinforce_plus_plus   # 原 grpo
--kl-coef 0.01                              # 原 0.0（必须非零，否则退化回 GRPO）
--normalize-advantages                      # 新增，必须
--kl-loss-coef 0.0                          # 保持（REINFORCE++ 的 KL 在 return 里）
# 去掉 --rewards-normalization（或保留，不影响 REINFORCE++）
# N_SAMPLES_PER_PROMPT 可降到 1（不需要 group）
```

不需要改的部分：rollout/采样逻辑、policy loss（[ppo_utils.py:124-148](slime/utils/ppo_utils.py#L124-L148)）、checkpoint、reward model 集成。

---

## Training Issues Compendium

以下按根因分类，每条给出现象、根因、解决方案。完整 traceback 见 `decoded/5-training-issues.md`。

### Parallel Configuration

**MoE + TP 未开 SP → ValueError**
MoE 层在同时开 TP 时若不开 Sequence Parallelism，序列维度通信冗余，Megatron 直接以 `ValueError` 拒绝。加 `--sequence-parallel`。

**层数与 PP 不整除 → AssertionError**
`Number of layers should be divisible by the pipeline-model-parallel size`。Qwen3.5 397B 共 60 层，PP=8 时无法整除——设 `--last-layer=4` 让最后一个 stage 少几层，或改 PP=4/5/6/10/12。

**World size 与 total_model_size 不整除**
`world size (8) is not divisible by total_model_size (32)`，以及 Placement Group 排序时的 `TypeError: '<' not supported between instances of 'str' and 'int'`。根因相同：Ray 集群有节点未连通。`ray status` 确认所有节点在线，重启机器等集群稳定。

**SGLang 启动 Qwen3-VL 时 vision tower TP 约束**
Driver 端只看到 `Server process terminated unexpectedly`，真实栈在 bootstrap 日志：`AssertionError: 16 is not divisible by 32`。Vision tower attention head 数固定 16，SGLang 用整个引擎 TP 世界大小切分（不受 `--enable-dp-attention` 影响），`ROLLOUT_ENGINE_SIZE=32` 时 `16 % 32 != 0`。解决：`ROLLOUT_ENGINE_SIZE` 缩至 ≤ 16，或换不带 vision tower 的纯文本 checkpoint。SGLang 启动失败的真实栈永远在 `worker-*.err` 或 nohup 日志里，driver 端的包装消息不能止步于此。

### Memory and Resource

**CUDA OOM（log-probs 路径）**
见 Log-Probs OOM 一节。两处高发：`compute_entropy_from_logits`（`vocab_parallel_logits - logits_max`）和 `cloned_logits = logits.clone()`。

**NCCL 共享内存不足**
`ncclSystemError: Error while creating shared memory segment /dev/shm/nccl-FCoKnB, error: No space left on device (28)`。`/dev/shm` 默认大小为物理内存一半，大规模训练时 NCCL 通信管道超出上限。`sudo mount -o remount,size=128G /dev/shm`。

**triton 首次编译耗时不齐 → NCCL timeout（伪装成 OOM）**
Driver 端看到 `ActorUnavailableError` + `killed by the OOM killer`，显存监控正常。真实链路：各 rank 不同时完成 triton cold cache 编译（方差约 10 分钟），超过 NCCL collective 默认 timeout 600s，watchdog 触发 `SIGABRT`，Ray 对任何 `SYSTEM_ERROR` 退出都附上 "killed by OOM killer" 的模板文字。

区分标志：日志有 `NCCL Watchdog caught collective operation timeout` 且各 rank 打了 `triton_bundler ... is not empty - skipping!`（**编译成功**的标志，不是错误）→ 本问题；有 `CompilationError` → 编译真的挂了。

解决：调大 NCCL timeout（30-60min），或把首跑的 `/tmp/triton_cache/<model>/` 分发到所有节点，后续命中缓存。

### Ray Cluster and Infrastructure

**残留 Placement Group 导致卡在 pg.ready()**
日志停在 `Creating placement group with N GPUs...` 不推进。`ray status` 显示 `0.0 used of X.0 reserved in placement groups` 但 `Pending Demands: (no resource demands)`。上次任务被 `kill -9` 或节点宕机，PG 以 `CREATED` 状态残留占用资源。

```bash
python -c '
import ray; ray.init(address="<RAY_ADDRESS>")
from ray.util.state import list_placement_groups
from ray.util.placement_group import remove_placement_group, PlacementGroup
from ray._raylet import PlacementGroupID
for pg in list_placement_groups():
    remove_placement_group(PlacementGroup(PlacementGroupID(bytes.fromhex(pg.placement_group_id))))
'
# 清不掉则：
ray stop --force && pkill -9 -f raylet && pkill -9 -f gcs_server
```

**wandb 初始化超时**
slime 在 `train.py` 最顶部、PG 创建之前初始化 wandb，超时时连 Ray 都还没起来。调通流程时：`export WANDB_MODE=disabled`；需要记录时：`export WANDB_MODE=offline` 后用 `wandb sync` 补传。

### Checkpoint and Model Loading

**全局形状不匹配 → CheckpointingException**
`Global shape mismatch for loaded (torch.Size([4096])) and expected ((2048,))`。脚本里 `source` 了错误大小的模型配置文件（如把 `Pro` 写成了 `Flash`）。

**OptimizerParamScheduler 迭代数不匹配**
`class input value 768000 and checkpoint value 1000000 for total number of iterations do not match`。加载时 `--lr-decay-iters` 与检查点不一致。三选一：对齐配置；加 `--override-opt_param_scheduler` 让当前值覆盖旧值；加 `--finetune` 或 `--no-load-optim` 只载权重。RL 续训常用第二种。

### Framework Regression Bugs

**`--offload-train` + save → `CUDA error: invalid argument`**（slime PR #1856 回归）
训练 80+ 步正常，每到 `save_interval` 必崩：`torch.AcceleratorError: CUDA error: invalid argument` 在 `dist_checkpointing.save` 的 `all_gather_object` 内部。调大 `save_interval` 只是推迟崩溃。

根因：PR #1856 改变了 offload 时序，`async_train()` 结束后自动 `torch_memory_saver.pause()` 把模型参数从 GPU 释放。但 `train.py` 里 `save(rollout_id)` 紧跟在 `async_train()` 之后——等 Megatron save 走到第一次 `all_gather_object` 时，模型已被 pause，CUDA 设备状态不一致。

解决：在 `offload_train=True` 分支里显式调用 `actor_model.offload()`，关掉 `train()` 内部的自动 pause。上游 issue：THUDM/slime#1886。

---

## Observability: wandb Metrics Reference

### rollout/ 前缀（rollout 阶段记录）

由 [data.py:log_rollout_data](slime/backends/megatron_utils/data.py) 聚合，`rollout_data` 的所有非 token/mask 字段都会被 log。

| wandb key | 来源 | 含义 | 是否经过 norm |
|---|---|---|---|
| `rollout/raw_reward` | `sample.get_reward_value()` | reward model 原始输出，不做任何 norm | ❌ |
| `rollout/rewards` | `_post_process_rewards()` 处理后 | grpo/gspo：组内减均值 ± 除 std；reinforce++：等于 raw_reward | 视 estimator |
| `rollout/advantages` | `compute_advantages_and_returns()` 后 | 若开了 `--normalize-advantages`，这里是 whiten 之后的值 | ✅ 若开了 |
| `rollout/returns` | 同上 | 折扣累积回报（grpo/reinforce++ 的 returns == advantages） | ❌ |
| `rollout/log_probs` | 当前 policy log prob | 负值，正常范围 (-0.5, 0) | — |
| `rollout/ref_log_probs` | ref model log prob | rollout_id=0 时应与 log_probs 一致 | — |
| `rollout/entropy` | 当前 policy 熵 | 正常范围 (0, 0.5) | — |

`rollout/rewards` 与 `rollout/raw_reward` 的区别：`raw_reward` 是 reward model 原始输出，`rewards` 是经 `_post_process_rewards()` 处理后的值——grpo/gspo 下两者可能相差一个均值和标准差；reinforce++ 下两者完全相同。

`rollout/advantages` 记录的是已经 whiten 后的值（如果开了 `--normalize-advantages`），无法从 wandb 反推原始 advantage。

### train/ 前缀（训练阶段记录）

每个 train step 结束后，[model.py:703-715](slime/backends/megatron_utils/model.py#L703-L715) 把 `loss_dict` 加 `train/` 前缀写入 wandb。

| wandb key | 含义 |
|---|---|
| `train/loss` | policy loss + entropy bonus，用于反向传播 |
| `train/pg_loss` | PPO clipped policy gradient loss |
| `train/entropy_loss` | 策略熵，鼓励探索的正则项 |
| `train/pg_clipfrac` | clip 触发比例，越高说明 ratio 越常撞到 clip 边界 |
| `train/ppo_kl` | 新旧策略 KL 散度，监控策略偏移 |
| `train/kl_loss` | 显式 KL 项（仅 `--use-kl-loss` 时），与 ppo_kl 不同 |
| `train/grad_norm` | 梯度范数，监控梯度爆炸 |

### Response Length: Two Different Counts

`rollout/response_lengths` 与 `rollout/response_len/mean` 指的不是同一件事：

- `rollout/response_lengths`：生成的**全部** token 数，包括 user 输入、observation tokens、assistant 回复
- `rollout/response_len/mean`：`sum(loss_mask)`，即**参与训练**的 token 数（只有 loss_mask=1 的部分）

Agent 场景或多轮对话中，两者比值可达 1.5~2× 甚至更高，是正常现象。若比值异常大，检查 observation tokens 的 `loss_mask` 是否被正确置零。

`raw_response_length/response_length_max` 和 `min` 需要开 `--log-multi-turn` 才会记录（[arguments.py:857-861](slime/utils/arguments.py#L857-L861)），用于排查是否有样本达到 `--rollout-max-response-len` 上限。
