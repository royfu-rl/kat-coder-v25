# DATA_FORMAT.md — 把你自己的数据接入 my-deepswe

这是 DeepSWE 数据浏览器（heatmap / tasks / trials）的一份**干净重写版**，零依赖、纯原生
JS + HTML + CSS。它和原站一样是**完全数据驱动**的：没有后端、没有 API，所有视图都从
`artifacts/` 目录下的静态 JSON 文件读取。

> **换数据 = 替换 JSON 文件。** 不需要改任何代码。

---

## 1. 怎么跑起来

因为用了 ES modules + `fetch`，必须用 HTTP 服务（不能直接双击 `index.html`，`file://`
会被 CORS 拦）。任选一种：

```bash
cd my-deepswe
python3 -m http.server 8731
# 打开 http://localhost:8731/
```

或 `npx serve .` / `php -S localhost:8731` 等任意静态服务器均可。

部署：把整个 `my-deepswe/` 目录原样丢到任何静态托管（GitHub Pages / S3 / Nginx）即可。

---

## 2. 目录结构

```
my-deepswe/
├── index.html              # 壳：header + 视图挂载点
├── styles.css              # 全部样式（亮/暗主题 token 与原站一致）
├── js/
│   ├── app.js              # 入口：主题切换、hash 路由、视图分发
│   ├── dom.js              # DOM 工具 + 极简 markdown/diff 渲染 + 路由
│   ├── data.js             # ★ 数据层：加载 artifacts + 所有 helper + heatmap 聚合引擎
│   ├── view_common.js      # 共享的页头 + 三个 tab
│   ├── view_heatmap.js     # Heatmap 视图
│   ├── view_tasks.js       # Tasks 列表 + 任务详情页
│   └── view_trials.js      # Trials 表格 + 单条 trial 详情页（轨迹查看器）
└── artifacts/              # ★★★ 你只需要动这个目录 ★★★
    ├── summary.json        # 站点名 + 计数（页头/页脚展示）
    ├── tasks.json          # 任务目录（Tasks 列表）
    ├── trials.json         # 主数据表（Heatmap + Trials 都来自这里）
    ├── release.json        # 指向「单条 trial 大文件」的存放位置
    ├── tasks/
    │   └── <task_id>.json  # 每个任务的详情（bundle 的文件：Dockerfile/patch/instruction…）
    └── trial-artifacts/    # 每条 trial 的轨迹/patch/日志（可放本地，也可放 CDN）
        └── <trial_name>/
            └── agent/trajectory.json
```

---

## 3. 数据契约（一个一个文件讲）

下面每个文件都给出：**它驱动什么** + **字段表** + **最小示例**。
标 `★` 的字段是视图实际会用到的；其余字段可省略（省略时该处显示 `—`）。

### 3.1 `artifacts/summary.json` — 站点元信息

驱动：页头品牌名、`<title>`、页脚计数。**可选**（缺失时用默认名 `my-deepswe`）。

```json
{
  "name": "DeepSWE",
  "generated_at": "2026-05-20T19:36:27Z",
  "counts": { "tasks": 113, "repositories": 91, "rollout_attempts": 7232, "trials": 8852 }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | ★ 站点名，显示在 header 和标题栏 |
| `counts.tasks` / `counts.trials` | number | 页脚展示的计数 |

---

### 3.2 `artifacts/tasks.json` — 任务目录

驱动：**Tasks 列表页**（搜索/语言筛选/卡片）。也给 Heatmap 提供任务标题。

```json
{
  "scope": "deep-swe",
  "n_tasks": 113,
  "rows": [
    {
      "id": "abs-module-cache-flags",
      "problem_title": "Harden module loading, cache introspection, and script flags",
      "display_description": "Harden ABS module resolution and caching…",
      "language": "go",
      "repository": "abs-lang/abs",
      "repository_url": "https://github.com/abs-lang/abs",
      "base_commit_hash": "cb1b3b671d0ee9fa9da9f7b02f86967953ffd10a",
      "prompt_characters": 2524
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `n_tasks` | number | 任务总数（列表页显示） |
| `rows[]` | array | 任务列表 |
| `rows[].id` | string | ★ **主键**。必须与 `trials.json` 的 `task_name`、以及 `tasks/<id>.json` 文件名一致 |
| `rows[].problem_title` | string | ★ 卡片标题 / heatmap 行标题 |
| `rows[].display_description` | string | 卡片副标题（可空） |
| `rows[].language` | string | ★ 语言筛选 + 卡片角标 |
| `rows[].repository` | string | ★ 卡片仓库名 |
| `rows[].repository_url` | string | 任务详情页链接（可空） |
| `rows[].base_commit_hash` | string | 任务详情页展示（可空） |

---

### 3.3 `artifacts/trials.json` — 主数据表（最重要）

驱动：**Heatmap**（客户端实时聚合）+ **Trials 表格**。一行 = 一次 (task, model) rollout。

```json
{
  "scope": "all",
  "n_trials": 14276,
  "rows": [
    {
      "trial_name": "abs-module-cache-flags__2EpkXD2",
      "task_name": "abs-module-cache-flags",
      "source": "deep-swe",
      "eval_scope": "full",
      "model": "minimax-m2-7",
      "reasoning_effort": null,
      "provider": "openrouter",
      "harness": "mini-swe-agent",
      "passed": false,
      "errored": false,
      "n_agent_steps": 186,
      "cost_usd": 0.78,
      "n_input_tokens": 10464327,
      "n_output_tokens": 46695,
      "peak_context_tokens": 99641,
      "trial_duration_seconds": 3112.9,
      "has_trajectory": true,
      "critique": null
    }
  ]
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `trial_name` | string | ★ | **主键**，唯一。用于详情页 URL + 关联 `trial-artifacts/<trial_name>/` |
| `task_name` | string | ★ | 关联 `tasks.json` 的 `id` |
| `source` | string | ★ | 数据集来源。`"deep-swe"` 或 `"swebenchpro"`（自定义也行） |
| `eval_scope` | string | ★ | `"full"`（进入主榜/Heatmap）或 `"cross-bench"`（进入「跨基准分析」视图）。**见下方说明** |
| `model` | string | ★ | 原始模型 id（前缀会被自动清洗，如 `moonshot/kimi-k2-6` → `kimi-k2.6`） |
| `reasoning_effort` | string\|null | ★ | 推理强度，如 `"high"` / `"max"` / `"xhigh"`，无则 `null`。Heatmap「Split effort」据此拆分 |
| `passed` | bool | ★ | 是否通过（绿 badge） |
| `errored` | bool | ★ | 是否报错（灰 badge `error`）。`passed=false && errored=false` 即 fail |
| `n_agent_steps` | number | | Steps 列 + heatmap「Avg steps」 |
| `cost_usd` | number | | Cost 列 + heatmap「Avg/Total cost」 |
| `n_input_tokens` | number | | heatmap「Avg input tokens」 |
| `n_output_tokens` | number | | heatmap「Avg output tokens」 |
| `peak_context_tokens` | number | | heatmap「Avg peak context」 |
| `trial_duration_seconds` | number | | heatmap「Avg duration」 |
| `provider` / `harness` | string | | trial 详情页展示 |
| `has_trajectory` | bool | | `false` 时详情页不去拉轨迹文件 |
| `critique` | object\|null | | 仅 cross-bench 用，见下 |

**关于 `eval_scope`：**
- `"full"` + `source === "deep-swe"` → 进入默认的 DeepSWE Heatmap 和 Trials 表。
- `"cross-bench"` → 进入「Cross-benchmark analysis」来源（顶部 Source 下拉切换），
  会额外显示 dataset 筛选和「轨迹分析（critique）」列。
- **如果你只有一个数据集**：把所有行设成 `eval_scope: "full"`、`source: "deep-swe"` 即可，
  跨基准那一套可以完全不管。

**`critique` 对象**（只有想用「跨基准/轨迹分析」时才需要）：

```json
"critique": {
  "rating": "good",            // "good"=验证器判定可信, "bad"=有异议
  "tag": "PASS_LEGITIMATE",    // 短标签，显示在表格里
  "feedback": "The trial is a legitimate pass…",  // 详情页长文本
  "model": "claude-opus-4-7"   // 做这次分析的模型
}
```

---

### 3.4 `artifacts/tasks/<task_id>.json` — 单个任务详情

驱动：**任务详情页**的 Verifier / Solution / Environment / task.toml 几个 tab。
文件名必须等于 `tasks.json` 里的 `id`。**可选**——没有就只显示 Trials tab。

```json
{
  "task_id": "abs-module-cache-flags",
  "n_files": 7,
  "files": [
    { "path": "instruction.md",          "language": "markdown",   "size": 2524, "content": "# …" },
    { "path": "environment/Dockerfile",   "language": "dockerfile", "size": 259,  "content": "FROM …" },
    { "path": "solution/solution.patch",  "language": "diff",       "size": 23719,"content": "diff --git …" },
    { "path": "solution/solve.sh",        "language": "shell",      "size": 104,  "content": "git apply …" },
    { "path": "tests/test.patch",         "language": "diff",       "size": 30643,"content": "diff --git …" },
    { "path": "tests/test.sh",            "language": "shell",      "size": 3868, "content": "set -e …" },
    { "path": "task.toml",                "language": "toml",       "size": 840,  "content": "[task] …" }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `files[].path` | ★ 决定它进哪个 tab。识别的路径：`instruction.md`、`environment/Dockerfile`、`solution/solution.patch`、`solution/solve.sh`、`tests/test.patch`、`tests/test.sh`、`task.toml` |
| `files[].content` | ★ 文件全文。`.patch`/`diff` 会按 +/- 着色，`instruction.md` 会按 markdown 渲染 |
| `files[].language` | 语法标签（展示用） |

> 路径名是约定俗成的。你也可以只放 `instruction.md` 一个文件，其余 tab 会显示「未打包」。

---

### 3.5 `artifacts/release.json` — 大文件存放位置

驱动：trial 详情页去**哪里**取轨迹/patch/日志。这是「数据源可换」的关键——
轨迹文件通常很大（单个几百 KB ~ 几 MB），所以单独放，可以放本地也可以放 CDN。

```json
{
  "release_id": "local",
  "artifact_base_url": "artifacts",
  "artifact_patterns": {
    "trajectory":   "trial-artifacts/{trial_name}/agent/trajectory.json",
    "model_patch":  "trial-artifacts/{trial_name}/artifacts/model.patch",
    "agent_log":    "trial-artifacts/{trial_name}/agent/agent-log.txt",
    "verifier_output": "trial-artifacts/{trial_name}/verifier/test-stdout.txt"
  }
}
```

| 字段 | 说明 |
|---|---|
| `artifact_base_url` | ★ 基础 URL。`"artifacts"`=本地相对目录；也可填 `"https://your-cdn.com"` |
| `artifact_patterns.trajectory` | ★ 轨迹路径模板，`{trial_name}` 会被替换。详情页据此拼出完整 URL |

最终请求 = `${artifact_base_url}/${trajectory_pattern}`，其中 `{trial_name}` 替换为具体 trial。
本仓库默认指向本地 `artifacts/trial-artifacts/`，已附带一个示例轨迹。

---

### 3.6 `artifacts/trial-artifacts/<trial_name>/agent/trajectory.json` — 单条轨迹

驱动：trial 详情页的**轨迹步骤查看器**。格式是 ATIF（Agent Trajectory Interchange Format）。

```json
{
  "schema_version": "ATIF-v1.7",
  "agent": { "name": "mini-swe-agent", "version": "2.2.8", "model_name": "minimax/minimax-m2.7" },
  "steps": [
    { "step_id": 2, "source": "user", "message": "Please solve this issue: …" },
    {
      "step_id": 3,
      "source": "agent",
      "message": "I'll start by analyzing the codebase…",
      "reasoning_content": "Let me start by…",
      "tool_calls": [
        { "function_name": "bash", "arguments": { "command": "find . -name '*.go'" } }
      ],
      "observation": { "results": [ { "content": "./repl/repl.go\n…" } ] }
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `steps[].source` | ★ `"user"` / `"agent"` / `"system"`（system 步骤会被跳过） |
| `steps[].message` | ★ 该步的文本 |
| `steps[].reasoning_content` | 推理过程（灰色引用块展示，可空） |
| `steps[].tool_calls[]` | 工具调用。`function_name` + `arguments.command`（或整个 arguments 对象） |
| `steps[].observation.results[].content` | 工具返回（>4000 字符会截断展示） |

> 轨迹是可选的：`trials.json` 里 `has_trajectory: false` 的 trial，详情页只显示指标，不去取文件。

---

## 4. 接入自己数据的最小步骤

1. **生成 `trials.json`**（最重要）：把你的每次 rollout 整理成一行，至少给齐
   `trial_name / task_name / source / eval_scope / model / reasoning_effort / passed / errored`。
   想要 heatmap 的成本/步数等着色，再补上对应数值字段。
   - 单数据集场景：所有行 `source:"deep-swe"`、`eval_scope:"full"` 即可。
2. **生成 `tasks.json`**：每个任务一行，`id` 必须与 trials 的 `task_name` 对得上。
3. **改 `summary.json`** 的 `name` 和 `counts`（仅展示用，可随意）。
4.（可选）**每个任务一个 `tasks/<id>.json`**：放 instruction / patch / Dockerfile，启用任务详情页。
5.（可选）**轨迹**：把 `trajectory.json` 放到 `trial-artifacts/<trial_name>/agent/` 下，
   或把 `release.json` 的 `artifact_base_url` 指向你的 CDN。
6. 起一个静态服务器，打开即用。

### 字段对应关系（务必对齐的三处主键）

```
tasks.json   rows[].id   ─┬─→  trials.json  rows[].task_name
                          └─→  tasks/<id>.json  文件名

trials.json  rows[].trial_name  ──→  trial-artifacts/<trial_name>/…  目录名
```

---

## 5. 一致性说明（和原站的对照）

本重写版的 heatmap 聚合引擎是对原站逻辑的**忠实复刻**，已用原始数据逐格验证：
排序、汇总列、Split-effort 拆分、各统计量的格式化（百分比 / `$x.xx` / `9m 45s` / `12k`）
与配色公式 `color-mix(in oklch, var(--foreground) (norm·88+8)%, transparent)` 均与原站一致。

几个内化的约定（改数据时不用管，了解即可）：
- 模型名清洗：去掉 `provider/` 前缀，把 `m2-7` 之类还原成 `m2.7`，去掉 `-preview` 后缀。
- Heatmap 行/列按总体通过率降序排，相同则按标题字典序。
- 通过率分母 = `n_trials − n_errors`（报错的 trial 不计入通过率分母）。
- Trials 表为性能考虑最多渲染前 2000 行（会有提示），缩小筛选即可看全部。

