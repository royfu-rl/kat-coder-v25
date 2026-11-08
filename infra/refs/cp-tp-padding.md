# CP / TP 与两层 padding——slime get_batch 的 token 切分

> 背景:Qwen3.5-35B-A3B,`TP=2 / CP=8`,`SEQUENCE_PARALLEL=1`,
> `MAX_TOKENS_PER_GPU=25600`,dynamic batch,`qkv_format=thd`(默认)。
> 代码:`slime/backends/megatron_utils/data.py:get_batch`(路径 A,thd 普通)
> 与 `slime/backends/megatron_utils/cp_utils.py:slice_with_cp`。

## 两个正交的并行维度

一张物理 GPU 同时有两个身份,互不相关:

- **CP(context parallel,本例 8)**:一条样本序列**按 token 切给 8 张卡**一起算,
  需要全局信息(attention)时卡间通信。解决"单卡放不下长序列"。
- **TP(tensor parallel,本例 2)**:同一份数据,**模型权重按 hidden / heads 切给 2 张卡**。
  开 `sequence_parallel` 后,LayerNorm / dropout / residual 这类无法按 hidden 切的算子,
  改成在**卡内部**沿序列维度切成 TP 份分步算(算一半 → 通信 → 算另一半)。

设备网格里每张卡的坐标是 `(tp_rank, cp_rank)`。`slice_with_cp` 只用 `cp_rank`,
所以它**只解决 CP 维度**;TP 维度要靠第二层 pad 保证能整除。

## token 预算怎么落到每卡 25600

```
MAX_TOKENS_PER_GPU = 25600                      # 每卡前向序列长度预算(你设的)
max_per_bin = 25600 × cp_size = 25600 × 8 = 204800   # dp_schedule.py:120,打包装箱上限
```

micro-batch 用 `first_fit_pack` 按**完整样本长度**装箱,每个 mbs 总长 ≤ 204800。
随后 `slice_with_cp` 把这个 mbs 按 CP=8 切,本卡拿走约 1/8 ≈ 25600。
框架用 `max_per_bin = max_tokens_per_gpu × cp_size` 这个乘法,把"打包上限"和
"每卡实得"绑定,让你设的 25600 直接等于每卡前向时的序列长度。

> 配套自洽:`MAX_TOTAL_LEN_TRUNCATE=204800` = `max_per_bin`。哪怕来一条顶到
> 天花板的 20 万 token 样本,它独占一个 mbs、切 8 卡后每卡正好 25600,不溢出。

## 两层 padding——各伺候一个并行轴

| 层 | 代码位置 | 粒度 | 对齐到 | 为什么 |
|---|---|---|---|---|
| 第一层 | `slice_with_cp` | **逐样本** | `2×CP`(CP=8→16, CP=16→32) | CP 的 zigzag 均分,每卡取对称 2 段 |
| 第二层 | cat 之后 | **整个 mbs(本卡部分)** | `TP×mult`(无开关,路径 A 永远执行) | TP 必须整除(SP 切序列正确性)+ mult 做 GEMM 对齐 / 抗显存碎片 / 收敛 shape |

> 这里写的是 **slime 公版**(实际跑的版本):第一层 `align_size = 2 * cp_size`,**不含 TP**
> ([cp_utils.py:205,210](slime/slime/backends/megatron_utils/cp_utils.py#L205-L210));第二层 `pad_size = TP × mult`,
> **无 `ENABLE_PACKED_TOKEN_PADDING` 门控、路径 A 必走**([data.py:107-109](slime/slime/backends/megatron_utils/data.py#L107-L109))。
> krl 版不同(第一层 `TP×CP×2`、第二层带门控且自带 `×2CP`),差异见文末「slime 公版 vs krl 版」一节。

### 关键:路径 A 是"先切 CP,后 cat,再 pad"

```python
# data.py:98-113(路径 A,thd 普通)
tokens = [slice_with_cp(t, ...) for t in tokens]   # ① 逐样本按 CP 切,每条只剩"本卡的 2 段"
cu_seqlens = ...                                    #    记录本卡各段边界
tokens = torch.cat(tokens)                          # ② cat 的是"本卡已切好的片段",不是全局序列
pad = (pad_size - tokens.size(0) % pad_size) % pad_size   # pad_size = TP×128 = 256
tokens = F.pad(tokens, (0, pad), value=pad_token_id)      # ③ pad 加在本卡这条流末尾
cu_seqlens = torch.tensor(cu_seqlens).cuda() * cp_size    #    局部长度 × cp 还原全局,供 attention 记账
```

- CP 切分在 **①** 就完成了。**②** cat 的是本卡专属数据,**③** 的 pad 是本卡局部的事,
  之后**不再经过 CP 切分**,所以"cat+pad 后切不齐 CP"的风险**在路径 A 不存在**。
- 各卡一致性天然成立:每条样本每个 cp_rank 拿到的都是等长的 `2×chunk_size`,
  cat 后 8 卡本地总长相等 → 同一条 pad 公式 → pad 后仍等长。

> 对比路径 B(`--allgather-cp`,DSA 模型,如 DeepSeek-V3.2 / GLM-5.1):顺序反过来,
> "cat 全局 → pad → 切 CP",所以它的 pad 基数必须含 cp(`cp_size × pad_size`)才切得齐。
> Qwen3.5 不在 DSA 白名单,且没开 allgather-cp,所以走路径 A。

## 为什么"已经是单卡了还要考虑 TP"

因为 TP+SP 的切分是**在这张卡内部、跑前向的过程中再发生一次**,不是发给别的卡:

- 这张卡 CP 维度上拿到了自己的 ≈25600 token;
- 但跑到 LayerNorm 等算子时,sequence parallel 把这 25600 **在卡内**沿序列切成 TP=2 份分步算;
- 要能整除切成 2 份,25600 必须是 TP 的倍数。这就是第二层 pad 带 TP 的唯一原因。
- `×128` 与正确性无关,纯性能:GEMM tiling 对齐 + 让 activation 显存块大小离散化、减少碎片。

## 带数字的完整例子(CP=8)

一个 mbs 装了 3 条样本,长度 `[100, 300, 500]`。

### 第一层:逐样本 slice_with_cp(对齐 2×CP=16)

`chunk_size = ceil(len / 16)`,全长 pad 到 `16 × chunk_size`,**本卡拿对称 2 段 = `2×chunk_size`**:

| 样本 | len | chunk_size = ceil(len/16) | pad 后全长 16×cs | 本卡拿 2×cs |
|---|---|---|---|---|
| A | 100 | 7  | 112 | 14 |
| B | 300 | 19 | 304 | 38 |
| C | 500 | 32 | 512 | 64 |

> zigzag:把每条切成 `2×8=16` 段,cp_rank=k 拿第 k 段 + 第 (15−k) 段(一头一尾),
> 8 张卡负载均衡(因果 attention 下靠后 token 上下文更长,对称取段才公平)。
> 校验样本 C:8 卡 × 64 = 512 = 全长 ✓。

### 第二层:cat 本卡片段 → pad 到 TP×128=256

```
本卡 cat 长度 = 14 + 38 + 64 = 116
pad = (256 - 116 % 256) % 256 = 140
最终本卡序列长度 = 116 + 140 = 256   # 256 的倍数,可被 TP=2 整除 → SP 切成 2×128
```

(真实 run 里一个 mbs 装的样本多得多,cat 长度在 25600 上下;pad 到 256 的倍数后
约为 25600 + 几百。25600 是目标,实际是"向上对齐到 256 倍数"的值,不会恰好等于。)

## ⚠️ `MAX_TOKENS_PER_GPU` 是软目标,不是每卡硬上限——每卡实际会**超出**

> 本节按 **slime 公版**(实际跑的版本)。两个 pad 都生效,且**第二层 `TP×mult` 是无开关、必走**的,
> 它才是每卡最终长度的主导。

`MAX_TOKENS_PER_GPU` 控的是**装箱判定用的「原始长度」上限 ÷ CP**,**不等于**每卡真实拿到的 token 数:

| | 装箱(决定一个 bin 塞哪些样本) | 每卡实得(切完 CP + 两层 pad) |
|---|---|---|
| 代码 | [data.py:353](slime/slime/backends/megatron_utils/data.py) → `get_minimum_num_micro_batch_size` ([utils/data.py:284](slime/slime/utils/data.py#L284)) | `slice_with_cp` 逐样本(对齐 `2×CP`)→ cat → pad 到 `TP×mult` |
| 用的长度 | **原始未 pad** `total_lengths` | 见下 |
| 判据 | `Σ length ≤ max_tokens_per_gpu × cp_size` | —— |

每卡最终长度经过两层向上取整,装箱判定只看原始长度,所以**每卡实得 ≥ 目标**:

```
本卡 cat 长度 = Σ_i ( ceil(len_i / (2·CP)) · 2 )          # 第一层:逐样本对齐 2·CP,本卡拿 padded/CP
每卡最终     = ceil(本卡 cat 长度 / (TP·mult)) · (TP·mult)  # 第二层:无条件对齐 TP·mult
```

### 主导项就是你说的 `TP×mult`(公版第二层,必走)

[data.py:107-109](slime/slime/backends/megatron_utils/data.py#L107-L109) 把本卡 cat 后的流**无条件**向上 pad 到
`pad_size = TP × mult` 的倍数,没有任何环境开关。于是:

```
每卡最终 token = MAX_TOKENS_PER_GPU 附近、向上取到 (TP×mult) 倍数的值
最坏多浪费    ≤ TP×mult − 1     （单个 mbs 的尾部 pad）
```

你的理解正确:**公版下每卡最多就是「向上对齐到 `TP×mult`」那一档**,封顶超出 < `TP×mult`。
本次设 `mult=512`、TP=2 → `pad_size = 1024`,即每卡长度被对齐到 **1024 的倍数**,尾部最多浪费 1023 token。

- small(CP=8, target 24576):24576 已是 1024 的倍数 → 若 cat 恰好 24576 则零尾 pad;实际 cat 多为 24576+几,会被推到 25600。
- big(CP=16, target 18432):18432 = 1024×18,同样落在档上。

### 还有一层更小的 CP 尾差(第一层,通常可忽略)

第二层之前,第一层 `slice_with_cp` 已对每条样本对齐到 `2×CP`(公版**不含 TP**,[cp_utils.py:205,210](slime/slime/backends/megatron_utils/cp_utils.py#L205-L210)),
本卡每条贡献是 `2`(=`2·CP/CP`)的倍数、每条最多浪费 `(2·CP−1)/CP < 2` token。装 `k` 条样本时这层至多带来 `~2k` 的零头,
但**它和最终长度都会被第二层 `TP×mult` 再向上吸收**,所以只要 `TP×mult` 不变,第一层的零头不改变「每卡落在哪个 1024 档」——
除非它恰好把 cat 长度顶过一个 1024 边界。结论:**每卡长度由 `TP×mult` 这一档决定,第一层只是档内的小扰动。**

## slime 公版 vs krl 版:pad 公式差异 → shape 档数 → 编译时间

⚠️ **最容易踩坑的一点**:第二层 pad 的公式两个仓库**不一样**,直接拿 krl 的数
套公版会算错 shape 档数。配套笔记见 `triton-cache.md`(那篇记的是 **krl** 版)。

### 公式差异

| | krl 版(`triton-cache.md` 记的) | **slime 公版(本 run)** |
|---|---|---|
| 第二层 pad 公式 | `pad_size = TP × mult × 2 × CP` | **`pad_size = TP × mult`**(不含 CP) |
| CP 怎么进 | 揉进 pad_size | 靠 `cu_seqlens × cp_size`,pad 不含 CP |
| `mult` 默认 | argparse `1`,rebench 脚本覆盖 `1024` | argparse 默认 **128** |
| pad 开关 | `ENABLE_PACKED_TOKEN_PADDING` 门控,默认关 | **无开关,路径 A 永远开** |

> 公版把 CP 的整除需求挪到了 `cu_seqlens × cp_size` 那一步(见上面 ③),
> 所以第二层 pad 只剩 TP 因子。krl 是把 CP 揉进 pad_size 一次解决。

### 为什么这影响 Triton 编译时间

Triton/FLA 的编译缓存按 **shape(序列长度)做 key**:每遇到一个没见过的长度,
就 autotune + 编译一次(几秒~几十秒/kernel)。**编译总开销 ∝ 不同 shape 的种数**:

```
shape 档数 ≈ ⌈每卡长度范围 / pad_size⌉ = ⌈max_tokens_per_gpu / pad_size⌉
```

pad_size 越大 → 桶越粗 → shape 种类越少 → 编译越少(但单 batch pad 浪费越大)。

### 同样 mult,两个仓库 pad_size 差 `2×CP` 倍(本例 16 倍)

`TP=2, CP=8, max_tokens_per_gpu=25600`:

| `mult` | **slime** pad_size = TP×mult | slime shape 档数 | krl pad_size = TP×mult×2CP | krl shape 档数 |
|---|---|---|---|---|
| 128(公版默认) | 256 | **~100 档** | 4096 | ~7 档 |
| 1024(krl 脚本值) | 2048 | ~13 档 | 32768 | **1 档**(撑满) |

**关键结论:**

- **公版默认 `mult=128` → pad_size 仅 256 → 多达 ~100 种 shape。** 预热期每种长度
  各编一次,autotune 次数远多于 krl,**冷启动编译时间显著更长**,但稳态 pad 浪费 <1%。
- **krl 同样 mult 下 pad_size 大 16 倍**(公式自带 `×2CP`)+ 脚本又把 mult 设 1024
  → shape 压到 1 档 → 编译零摩擦、CUDA graph / 离线 cache 命中率 100%,代价是
  每 batch 浪费 ~28% forward 算力(本地 25600 撑到 32768)。

### 想在公版复现 krl 的"少 shape、快编译"

公版没有 `×2CP`,要达到 krl 的 pad_size 得把 mult 调大补回来:

- 想要 1 档(全撑到 25600 倍数):`pad_size ≥ 25600` → `mult ≥ 12800`。
- 折中(~13 档):`mult=1024` → pad_size 2048,稳态浪费 ~4%。

权衡同 `triton-cache.md §2.3`:**长训练**稳态省的算力通常远超一次预热,小~中 mult
更划算;**极短训练 / shape 持续新增**才值得大 pad 锁档 + 配离线 cache。
调旋钮:`--data-pad-size-multiplier`(公版默认 128)。

## 一句话总结

- **CP 切的是**"哪 8 张卡分这条样本"——`slice_with_cp` 逐样本对齐 16,先切后 cat。
- **TP 切的是**"本卡这段序列,SP 时再被卡内的 2-way 分着算"——cat 后对齐 256。
- 一张物理 GPU 站在 `(tp_rank, cp_rank)` 交叉点,两个维度的整除约束都得满足,故两层 pad。
- 路径 A 先切 CP 后 pad,pad 落在本卡局部流上,不破坏 CP;`MAX_TOKENS_PER_GPU` 就是
  cat 后本卡的目标 token 数。

