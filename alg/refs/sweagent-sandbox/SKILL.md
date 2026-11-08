---
name: sweagent-sandbox
description: Use this skill to audit swe-agent (swe-rex) RL training trajectories for SANDBOX problems and then review the findings for misjudgments. Triggers on requests like "扫描这批 swe-agent 轨迹找沙箱问题", "audit these .traj files for sandbox bugs", "复核扫描结果有没有把模型问题当成沙箱问题", "给我每类沙箱问题的 x/N 计数". Targets swe-agent .traj datasets classified by the 入口/会话/出口/未分类 scheme. Do NOT use for Claude-Code / miniagent request.json datasets (those use a separate process.run scanner).
---

# swe-agent Sandbox Audit

A three-stage pipeline that scans swe-agent trajectory logs for **sandbox
problems** (execution-environment defects, not model mistakes), then reviews its
own findings to strip out misjudgments, then reports per-category `x/N` counts.

Classification follows the settled scheme in `swe-rubric.md` — a pipeline by
link position:

```
模型意图(Action) →  入口  →  会话  →  出口  → 模型观测到(Observation)
                   命令转写   状态延续   结果回传
```

- **入口·命令转写失真** — intent is fine, but the command string is corrupted
  (quoting/escape, multi-line join, heredoc, encoding) before it ever runs.
- **会话·步间状态延续** — each command is fine alone, but the persistent
  shell/PTY leaks state across steps (cwd/env, unclosed heredoc, PTY/readline,
  background process, pager).
- **出口·结果回传失真** — the command runs correctly, but its output is distorted
  on the way back (truncation, stdout/stderr interleave, control-sequence leak).
- **未分类** — catch-all for sandbox-related anomalies that don't fit the three.

The dominant audit error is **attributing a model's own mistake to the sandbox**
(e.g. the model wrote a bad sed, a build tool is missing, a test failed because
the fix was wrong). Stage 2 exists specifically to catch this.

## When to use

- User points at a swe-agent dataset (dir of `eval_N/.../.../*.traj`) and wants
  sandbox problems found.
- User wants the stage-1 scan results re-checked for false positives.
- User wants the final per-category `x/N` caliber table.

## When NOT to use

- Claude-Code / miniagent trajectories (OpenAI-style `*_request.json`) — that
  format needs the separate process.run scanner, not this one.
- Auditing **model behavior** problems (this skill is sandbox-only).

## Prerequisites

Set before running any stage:

| Env | Required | Meaning |
|-----|----------|---------|
| `ANTHROPIC_AUTH_TOKEN` | yes | API token (or `ANTHROPIC_API_KEY` for native `sk-ant-`) |
| `ANTHROPIC_BASE_URL` | for relay | override endpoint; switches auth to `Bearer` |
| `CLAUDE_MODEL` | no | default `claude-opus-4-7` |
| `CONCURRENCY` | no | parallel API workers, default 4 |

Confirm `ANTHROPIC_AUTH_TOKEN` is exported before launching — the scripts abort
without it. `scripts/` is resolved relative to this SKILL.md.

## The 3-stage workflow

Run the stages in order. Each writes a file the next stage reads; **stage 1 and
stage 2 are resumable** (re-running skips already-done evals/bugs), so a killed
run can be resumed by re-invoking with the same paths.

### Stage 1 — Scan (concurrent)

Confirm `DATASET_DIR` (root that directly contains `eval_0`, `eval_1`, …) and
choose an `OUTPUT_FILE`. Then:

```bash
DATASET_DIR=/path/to/dataset \
OUTPUT_FILE=/path/to/scan-data.md \
CONCURRENCY=4 \
bash scripts/scan.sh
```

Each eval is audited against the embedded 入口/会话/出口/未分类 prompt (the full
"不是 swe-rex bug" exclusion list is baked in to keep precision high). One
`## eval_N <instance_id>` section per eval, each with 样本概览 / 风险面分析 /
Bug 发现 (`NO_BUG` or `BUG:` blocks).

After it finishes, report how many evals were scanned and the raw bug count
(`grep -c '^BUG:' "$OUTPUT_FILE"`). **Stop and let the user look before stage 2.**

### Stage 2 — Evaluate (false-positive review)

Re-check every `BUG:` block from stage 1. Each bug is judged **from its block
text alone** — the goal is catching model/environment problems misattributed as
sandbox.

```bash
OUTPUT_FILE=/path/to/scan-data.md \
REVIEW_FILE=/path/to/review.md \
CONCURRENCY=4 \
bash scripts/evaluate.sh
```

Each bug gets a verdict: `真沙箱` / `误判(模型问题)` / `误判(合法环境/行为)` /
`存疑`, plus a one-line reason, written to `REVIEW_FILE`. Report how many were
upheld as `真沙箱` vs flagged as misjudgments.

### Stage 3 — Summarize (x/N caliber table)

```bash
python3 scripts/summarize.py /path/to/scan-data.md /path/to/review.md /path/to/summary.md
```

Keeps only `真沙箱` bugs, dedups by eval within each category (an eval counts at
most once per category), and prints the caliber table: per-category `x/N` plus
`去重合计 = 至少含一个真沙箱问题的 trajectory`. `N` is the number of evals actually
scanned, read from the scan file — never hardcoded.

Present **only the final conclusion** — the per-category counts and the deduped
total. The user wants the result, not the derivation.

> RAW mode (no review file): `python3 scripts/summarize.py scan-data.md` counts
> every bug block unfiltered — useful for a quick parser sanity check, not for
> the final number.

## Important behaviors

- **保守原则·宁漏不错**: when a bug is ambiguous between sandbox and model, it
  does NOT count as a real sandbox problem. Both the scan prompt and the review
  prompt enforce this; do not override it when reporting.
- **Sandbox-only**: never fold model-behavior problems into the counts. If the
  user also wants behavior problems, that's a separate pass.
- **Don't hardcode N**: the denominator is whatever stage 1 actually scanned.
- **Stop between stages 1 and 2** so the user can sanity-check the raw findings
  before the review pass spends more tokens.
- Reference: the human-readable classification rubric is bundled at
  `references/swe-rubric.md` (relative to this SKILL.md). The scan/evaluate
  prompts already embed the scheme — this file is for humans, not the scripts.

