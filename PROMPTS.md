# 本次工作 Prompt 记录

对 `kat-coder-rl-blog`（`build.py` 从 markdown 源生成单文件 `index.html` 的技术报告站点）的一轮修复。下面按用户实际下达的指令整理。

---

## 第 1 轮 Prompt

> 我提到的所有 skill 都重定向到 `C:\Users\roy\Desktop\Workspace\learner\6-skill`。

**Task 1 — 修复无法渲染的公式**
页面里若干 `$$…$$` / `$…$` 公式渲染失败，示例（均来自 credit-assignment 文档）：
- `\rho^{i,k,t}=\frac{\pi_\theta(y^{i,k,t}\mid h^{i,k},y^{i,k,<t})}...`
- `\rho^{i,t}=\pi_\theta(y^{i,t}\mid h^{i,<t})...`
- 以及 IPA 的 `\rho^{i,k}=\Bigl(\prod_{t\in c_k}...\Bigr)^{1/|c_k|}` 等

**Task 2 — 删除 overview 冗余元素**
- 删除横幅：`本页为总览，各模块的详细设计见下方 详情 → 链接。`
- 删除四条 `详情 → env/env.md | infra/infra.md | alg/alg.md | eval/eval.md`（上方 Tab 已可跳转）

**Task 3 — arXiv 按钮旁增加跳转链接**
除论文外还有宣发稿，新智元微信公众号：
`https://mp.weixin.qq.com/s/79XLhfEe7rdGhqR1M2Xqkg`（找新智元 logo，作为 demo 展示）

**Task 4 — 补齐缺失图片**
1. `flash-bench.png` — KAT-Coder-V2.5-Flash benchmark：由 `grouped-bars-green` skill 生成的 HTML 截图，用户直接提供 HTML 源码，要求内嵌
2. `pro-bench.png` — KAT-Coder-V2.5-Pro benchmark：HTML 源码在 skill 内
3. `rl-infra.png` — 原件丢失，后补：Agentic RL Infrastructure — Any Harness Adapter / Gateway Server 图已补入 `assets/figures/`
4. `failure-modes.png` — Failure modes by model（SWE-Bench Pro / DeepSWE）：给出失败模式分类表，用 `stacked-bar-tabs` skill 生成 HTML
5. `autobuilder.png` — 原件丢失，后补：Scalable PR-to-Executable-Instance Generation Framework 图已补入 `assets/figures/`
6. `compare-tabs.png` — 对标行业头部模型（分语言 / 仓库类型 / 任务类型 / 改动规模），用 `grouped-bars-green` + `stacked-bar-tabs` 两个 skill 协同设计（可用 bar 切换），给出四组数据

---

## 第 2 轮 Prompt

1. `Failure-mode taxonomy by model` 这个 HTML 延续 `Evaluation — Auto-Eval 与轨迹治理` 即可。
2. `对标行业头部模型…细分横评` 的 HTML 还没做/没展示出来 —— 需排查。
3. 全部弄好后 git commit 以及 push。
4. 梳理本次工作的 prompt，存储下来（本文件）。

---

## 实际改动小结

- **公式**：`renderDoc` 把还原的 `$$…$$` 直接塞进 `innerHTML`，浏览器把 `<t` 当成 HTML 标签，`$$` 定界符被拆散、KaTeX 找不到。修复：还原前对 `< > &` 做 HTML 转义（`renderMathInElement` 读 `textContent`，会自动解码回正确 LaTeX）。
- **build.py（潜在 bug）**：`os.path.relpath` 在 Windows 产生反斜杠路径，浏览器不认；改为统一输出正斜杠。
- **overview.md**：删除总览横幅与四条「详情 →」。
- **template.html**：arXiv 按钮旁加「新智元报道」链接；showcase 的 flash / pro 由 `<img>` 改为 `<iframe>` 内嵌图表。
- **eval.md**：failure-modes、compare-tabs 由缺失 PNG 改为 `<iframe>` 内嵌图表。
- **新增文件**：`assets/figures/flash-bench.html`、`assets/figures/pro-bench.html`、`eval/charts/failure-modes.html`（compare-tabs.html 已存在且数据正确，未重做）。
- **vendored 文档迁移**：`alg.md` / `infra.md` 里三处指向 macOS 绝对路径的链接（credit-assignment、sweagent-sandbox、cp-tp-padding），而 `build.py` 每次清空 `vendor/`，Windows 重建会静默丢掉含全部公式的 credit-assignment 文档；已把三份文档迁入 `alg/refs/`、`infra/refs/` 并改为相对路径。

