# Model Releases: GPT, Claude, Kimi

比较三家模型之前，先固定一个事实：能力跃迁发生在版本号的小数点后。GPT 从 5.0 到 5.6、Claude 从 4 到 5 代、Kimi 从 K1.5 到 K3——真正拉开差距的不是大版本首发，而是随后半年内密集的小版本迭代。下面按发布日期排列每一次公开发版，模型名直接指向该版本的官方技术报告、system card 或发布公告，用同一套坐标读三条演进线。

三家的报告习惯不同，读表前先记住：OpenAI 从 GPT-4 起不再发 arXiv 论文，改用 system card；Anthropic 从不发模型 arXiv，只出 model card 与公告；Moonshot 是三家里唯一持续在 arXiv 发技术报告的，但 K2.6 之后也转向仓库内 PDF。arXiv 论文链接统一收在文末 References。

## OpenAI GPT

下表按发布日期排序。GPT-4 之前的模型有 arXiv 论文或官方 PDF，之后一律只有公告页与 system card。

| 发布日期 | 模型 | 能力要点 |
|---|---|---|
| 2018/06/11 | [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 首个生成式预训练 Transformer，约 117M 参数，预训练加微调范式奠基 |
| 2019/02/14 | [GPT-2](https://openai.com/index/better-language-models/) | 1.5B 参数，出于安全分阶段放出，零样本生成 |
| 2020/05/28 | [GPT-3](https://arxiv.org/abs/2005.14165) | 175B 参数，few-shot 上下文学习 |
| 2022/01/27 | [InstructGPT](https://openai.com/index/instruction-following/) | RLHF 对齐指令跟随，GPT-3.5 的基础 |
| 2022/11/30 | [ChatGPT (GPT-3.5)](https://openai.com/index/chatgpt/) | 对话式 RLHF 模型，引爆消费级采用 |
| 2023/03/14 | [GPT-4](https://openai.com/index/gpt-4-research/) | 多模态输入，能力大幅跃升，架构不公开 |
| 2023/11/06 | [GPT-4 Turbo](https://openai.com/index/new-models-and-developer-products-announced-at-devday/) | 128K 上下文，更便宜，知识截止刷新 |
| 2024/05/13 | [GPT-4o](https://openai.com/index/hello-gpt-4o/) | 原生多模态（文本 / 音频 / 视觉），实时语音 |
| 2024/07/18 | [GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) | 低成本小模型，替代 GPT-3.5 Turbo |
| 2024/09/12 | [o1-preview / o1-mini](https://openai.com/index/introducing-openai-o1-preview/) | 首批推理模型，推理时长链式思考 |
| 2024/12/05 | [o1](https://openai.com/index/o1-and-new-tools-for-chatgpt-users/) | 完整版 o1，数学 / 代码 / 科学更强 |
| 2025/01/31 | [o3-mini](https://openai.com/index/openai-o3-mini/) | 低成本推理模型，推理力度可调 |
| 2025/02/27 | [GPT-4.5 (Orion)](https://openai.com/index/introducing-gpt-4-5/) | 最大的非推理模型，后于 2025/07 下线 |
| 2025/04/14 | [GPT-4.1](https://openai.com/index/gpt-4-1/) | 偏代码，最高 1M token 上下文，仅 API |
| 2025/04/16 | [o3 / o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/) | 前沿推理加完整工具使用 |
| 2025/08/07 | [GPT-5](https://openai.com/index/gpt-5-system-card/) | 统一模型加快 / 慢模式路由 |
| 2025/11/12 | [GPT-5.1](https://openai.com/index/gpt-5-1/) | Instant / Thinking / Codex 家族，11/19 补 Codex-Max 与 Pro |
| 2025/12/11 | [GPT-5.2](https://openai.com/index/introducing-gpt-5-2/) | GPT-5.1 的增量继任 |
| 2026/03/05 | [GPT-5.4](https://openai.com/index/introducing-gpt-5-4/) | 主线 GPT-5.x 迭代，前置版本为 GPT-5.2 |
| 2026/04/23 | [GPT-5.5 (Spud)](https://deploymentsafety.openai.com/gpt-5-5/) | 代码 / 科研 / 知识工作准确性提升，含 GPT-5.5 Pro |
| 2026/07/09 | [GPT-5.6](https://deploymentsafety.openai.com/gpt-5-6/) | 三变体 Sol / Terra / Luna，官方称最强代码与网络安全模型 |

沿时间线读，OpenAI 的演进分四段，每一段换一次产品逻辑。

- **2018–2020 — 规模定律奠基**：GPT-1 立起预训练加微调范式，GPT-2 用 1.5B 参数验证零样本生成，GPT-3 把参数推到 175B 并证明 few-shot 上下文学习可行。这三代都有公开论文，是唯一能读到架构细节的时期。

- **2022–2023 — 对齐与多模态**：InstructGPT 用 RLHF 把模型对齐到指令跟随，直接催生 ChatGPT 的消费级爆发；GPT-4 加入多模态输入并大幅提升能力，但从此不再公开架构，arXiv 论文也止步于此。

- **2024 — 效率与推理双线**：GPT-4o 做原生多模态与实时语音，GPT-4o mini 压成本替代 3.5 Turbo；o1 系列开辟推理模型这条独立产品线，用推理时的长链式思考换数学与代码能力。

- **2025–2026 — 版本号内卷**：GPT-5 首次用统一模型加路由整合快慢两种模式，随后 5.1、5.2、5.4、5.5、5.6 在不到一年内密集迭代。到 GPT-5.6 分裂出 Sol / Terra / Luna 三个变体，能力档位从单一模型变成一个梯队。当前最新主线是 GPT-5.6（2026/07/09）。

## Anthropic Claude

下表按发布日期排序。Anthropic 从不发模型 arXiv 论文，只出公告与 model card。Claude Code（命令行 agent）随 Claude 3.7 Sonnet 一同发布。

| 发布日期 | 模型 | 能力要点 |
|---|---|---|
| 2023/07/11 | [Claude 2](https://www.anthropic.com/news/claude-2) | 100K 上下文，编码与长文档处理 |
| 2023/11/21 | [Claude 2.1](https://www.anthropic.com/news/claude-2-1) | 200K 上下文，降低幻觉 |
| 2024/03/04 | [Claude 3 (Haiku/Sonnet/Opus)](https://www.anthropic.com/news/claude-3-family) | 三档模型分级，多模态视觉输入 |
| 2024/06/20 | [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) | 代码能力大涨，Artifacts 功能 |
| 2024/10/22 | [Claude 3.5 Sonnet (new) / 3.5 Haiku](https://www.anthropic.com/news/3-5-models-and-computer-use) | 首推 computer use，代理操作电脑 |
| 2025/02/24 | [Claude 3.7 Sonnet + Claude Code](https://www.anthropic.com/news/claude-3-7-sonnet) | 首个混合推理模型，Claude Code 命令行 agent 同期发布 |
| 2025/05/22 | [Claude 4 (Opus 4 / Sonnet 4)](https://www.anthropic.com/news/claude-4) | 扩展思考加工具使用，SWE-bench 领先 |
| 2025/08/05 | [Claude Opus 4.1](https://www.anthropic.com/news/claude-opus-4-1) | Opus 4 增量升级，代理编码提升 |
| 2025/09/29 | [Claude Sonnet 4.5](https://www.anthropic.com/news/claude-sonnet-4-5) | 长程代理任务，编码标杆 |
| 2025/10/15 | [Claude Haiku 4.5](https://www.anthropic.com/news/claude-haiku-4-5) | 小模型逼近前代 Sonnet，低延迟低成本 |
| 2025/11/24 | [Claude Opus 4.5](https://www.anthropic.com/news/claude-opus-4-5) | 旗舰 Opus 回归，前沿代码与推理 |
| 2026/02/05 | [Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) | Opus 4.5 增量迭代 |
| 2026/02/17 | [Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6) | Sonnet 中端更新 |
| 2026/04/16 | [Claude Opus 4.7](https://www.anthropic.com/news/claude-opus-4-7) | Opus 线继续迭代 |
| 2026/05/28 | [Claude Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8) | Claude 5 代之前的最后一版 Opus 4.x |
| 2026/06/09 | [Claude Fable 5 / Mythos 5](https://www.anthropic.com/news/claude-fable-5-mythos-5) | Claude 5 代首发，Fable 5 是 Mythos-class 的通用安全版，几乎全 benchmark SOTA |
| 2026/06/30 | [Claude Sonnet 5](https://www.anthropic.com/news/claude-sonnet-5) | Claude 5 代中端 |
| 2026/07/24 | [Claude Opus 5](https://www.anthropic.com/news/claude-opus-5) | 逼近 Fable 5 智能、成本约减半 |

沿时间线读，Claude 的演进分三段，命名体系在第三段发生突变。

- **2023–2024 上半年 — 长上下文与分级**：Claude 2 立起 100K 上下文优势，2.1 推到 200K；Claude 3 用 Haiku / Sonnet / Opus 三档做能力分级并加入多模态；3.5 Sonnet 让代码能力成为 Claude 的招牌。

- **2024 下半年–2025 — 代理化转向**：3.5 Sonnet(new) 首推 computer use，让模型直接操作电脑；3.7 Sonnet 成为首个混合推理模型，并同期发布 Claude Code 这一命令行 agent，把 Claude 推向编码代理场景；Claude 4 与随后的 Opus 4.1、Sonnet 4.5、Haiku 4.5、Opus 4.5 在 SWE-bench 等编码基准上持续领先。这一时期 Opus 4.x 密集迭代到 4.8。

- **2026 — Claude 5 代与命名突变**：Claude 5 代不再沿用 Opus / Sonnet / Haiku 命名，改用 Fable 5、Mythos 5 这类新序列。Fable 5（2026/06/09）是 Mythos-class 的通用安全版，几乎在所有测试基准上 SOTA；Mythos 5 是更高能力档，因出口管制一度受限。随后 Sonnet 5、Opus 5 回归旧命名，Opus 5（2026/07/24）逼近 Fable 5 智能而成本约减半。当前 Claude Code 默认使用 Claude 5 代模型。

## Moonshot Kimi

下表按发布日期排序。Kimi 是三家里唯一持续在 arXiv 发技术报告的，k1.5 / K2 / K2.5 均有 arXiv 论文；早期产品里程碑与 K2.6 之后的模型转为官网或仓库内 PDF。

| 发布日期 | 模型 | 能力要点 |
|---|---|---|
| 2023/10/09 | [Kimi Chat](https://www.moonshot.cn/) | 首发，约 20 万字上下文，当时最长之一 |
| 2024/03/18 | [Kimi Chat (200 万字)](https://www.moonshot.cn/) | 上下文扩到约 200 万字，长上下文里程碑 |
| 2025/01/20 | [Kimi k1.5](https://github.com/MoonshotAI/Kimi-k1.5) | RL 扩展的多模态推理模型，长短 CoT，部分基准对标 o1 |
| 2025/07/11 | [Kimi K2 (Base / Instruct)](https://moonshotai.github.io/Kimi-K2/) | 1T MoE（32B 激活），开源权重，MuonClip 优化器，偏代理与编码 |
| 2025/09/09 | [Kimi K2-Instruct-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905) | 上下文 128K→256K，代理编码大幅提升，SWE-bench Verified 69.2 |
| 2025/11/06 | [Kimi K2 Thinking](https://moonshotai.github.io/Kimi-K2/thinking.html) | 开源思考 / 代理模型，1T MoE，256K 上下文，原生 INT4 QAT，HLE 44.9（带工具） |
| 2026/02/02 | [Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5) | 原生多模态（文 / 图 / 视频），1T MoE，256K 上下文，引入 Agent Swarm |
| 2026/04 | [Kimi K2.6](https://kimi.com/blog/kimi-k2-6) | 多模态代理模型，图 / 视频输入，256K 上下文，自我纠错增强 |
| 2026/06 | [Kimi K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) | 偏代码的代理模型，强制思考模式，思考 token 减约 30% |
| 2026/07/16 | [Kimi K3](https://kimi.com/blog/kimi-k3) | 旗舰 2.8T MoE（104B 激活，896 专家），KDA + Gated MLA，原生视觉，1M 上下文 |

沿时间线读，Kimi 的演进分三段，从长上下文单点突破走到万亿参数开源旗舰。

- **2023–2024 — 长上下文单点突破**：初代 Kimi Chat 用约 20 万字上下文切入市场，2024 年扩到约 200 万字。这一时期没有技术报告，靠产品化的超长上下文建立认知。

- **2025 — RL 推理与开源 MoE**：k1.5 是首个有 arXiv 报告的版本，用 RL 扩展多模态推理并对标 o1；K2 把架构推到 1T MoE、开源权重、自研 MuonClip 优化器，明确转向代理与编码；随后 K2-0905 把上下文翻到 256K，K2 Thinking 加入原生 INT4 量化与思考模式。这一年 Kimi 从追随者变成开源前沿。

- **2026 — 多模态代理与万亿旗舰**：K2.5 加入原生多模态与 Agent Swarm，是这一代唯一有 arXiv 论文（2602.02276）的版本，后续 K2.6、K2.7-Code 复用同一份报告；K3（2026/07/16）把参数推到 2.8T MoE、104B 激活、896 专家，用 KDA + Gated MLA 架构与原生视觉支持 1M 上下文，号称首个开放的 3T 级模型。K3 的技术报告为仓库内 PDF，未上 arXiv。

---

# References

**OpenAI GPT**
- GPT-3 — [arXiv 2005.14165](https://arxiv.org/abs/2005.14165) ｜ GPT-4 — [arXiv 2303.08774](https://arxiv.org/abs/2303.08774) ｜ InstructGPT — [arXiv 2203.02155](https://arxiv.org/abs/2203.02155)
- GPT-5 System Card — [openai.com](https://openai.com/index/gpt-5-system-card/)
- GPT-5.5 System Card — [deploymentsafety.openai.com](https://deploymentsafety.openai.com/gpt-5-5/) ｜ GPT-5.6 System Card — [deploymentsafety.openai.com](https://deploymentsafety.openai.com/gpt-5-6/)

**Anthropic Claude**（无模型 arXiv 论文，均为官方公告 / model card）
- Claude 3.7 Sonnet + Claude Code — [anthropic.com/news](https://www.anthropic.com/news/claude-3-7-sonnet)
- Claude 4 — [anthropic.com/news](https://www.anthropic.com/news/claude-4)
- Claude Fable 5 / Mythos 5 — [anthropic.com/news](https://www.anthropic.com/news/claude-fable-5-mythos-5)
- Claude Opus 5 — [anthropic.com/news](https://www.anthropic.com/news/claude-opus-5) ｜ Claude Sonnet 5 — [anthropic.com/news](https://www.anthropic.com/news/claude-sonnet-5)

**Moonshot Kimi**
- Kimi k1.5 — [arXiv 2501.12599](https://arxiv.org/abs/2501.12599) ｜ [GitHub](https://github.com/MoonshotAI/Kimi-k1.5)
- Kimi K2 — [arXiv 2507.20534](https://arxiv.org/abs/2507.20534) ｜ [博客](https://moonshotai.github.io/Kimi-K2/)
- Kimi K2.5 — [arXiv 2602.02276](https://arxiv.org/abs/2602.02276) ｜ [GitHub](https://github.com/MoonshotAI/Kimi-K2.5)
- Kimi K2 Thinking — [博客](https://moonshotai.github.io/Kimi-K2/thinking.html) ｜ Kimi K3 — [博客](https://kimi.com/blog/kimi-k3)

