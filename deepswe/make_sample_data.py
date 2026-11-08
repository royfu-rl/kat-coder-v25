#!/usr/bin/env python3
"""
make_sample_data.py — 生成一份最小的可用示例数据，演示「换数据」流程。

运行后会在 ./sample_artifacts/ 下生成 summary.json / tasks.json / trials.json /
release.json，结构与 DATA_FORMAT.md 完全一致。想试效果就把它们拷进 artifacts/：

    python3 make_sample_data.py
    cp -r sample_artifacts/* artifacts/        # 覆盖示例数据
    python3 -m http.server 8731                # 打开 http://localhost:8731/

这里不用任何随机数，纯确定性生成，方便对照。
"""
import json, os, math

OUT = "sample_artifacts"
os.makedirs(f"{OUT}/tasks", exist_ok=True)

MODELS = [
    ("my-model-pro", "high"),
    ("my-model-pro", "low"),
    ("my-model-mini", None),
    ("baseline-7b", None),
]
TASKS = [
    ("fix-login-bug",   "Fix the broken login redirect",      "python", "acme/webapp"),
    ("add-csv-export",  "Add CSV export to the report page",  "typescript", "acme/dashboard"),
    ("optimize-query",  "Optimize the slow analytics query",  "go", "acme/api"),
]

# ---- tasks.json ----
tasks_rows = [{
    "id": tid, "problem_title": title, "display_description": title + ".",
    "language": lang, "repository": repo,
    "repository_url": f"https://github.com/{repo}",
} for (tid, title, lang, repo) in TASKS]
json.dump({"scope": "demo", "n_tasks": len(tasks_rows), "rows": tasks_rows},
          open(f"{OUT}/tasks.json", "w"), indent=2)

# ---- trials.json: 4 rollouts per (task, model) ----
trial_rows = []
for ti, (tid, title, lang, repo) in enumerate(TASKS):
    for mi, (model, effort) in enumerate(MODELS):
        for k in range(4):
            # deterministic pass pattern: stronger models pass more
            passed = (k < (4 - mi - (1 if effort == "low" else 0)))
            errored = (mi == 3 and k == 3)  # baseline occasionally errors
            if errored:
                passed = False
            name = f"{tid}__{model}-{effort or 'na'}-{k}"
            trial_rows.append({
                "trial_name": name,
                "task_name": tid,
                "source": "deep-swe",
                "eval_scope": "full",
                "model": model,
                "reasoning_effort": effort,
                "provider": "demo",
                "harness": "demo-agent",
                "passed": passed,
                "errored": errored,
                "n_agent_steps": 40 + ti * 10 + mi * 5 + k,
                "cost_usd": round(0.5 + ti * 0.3 + mi * 0.2 + k * 0.05, 4),
                "n_input_tokens": 1_000_000 + ti * 200_000 + k * 10_000,
                "n_output_tokens": 30_000 + mi * 5_000,
                "peak_context_tokens": 80_000 + ti * 5_000,
                "trial_duration_seconds": 600 + ti * 120 + k * 30,
                "has_trajectory": False,
                "critique": None,
            })
json.dump({"scope": "all", "n_trials": len(trial_rows), "rows": trial_rows},
          open(f"{OUT}/trials.json", "w"), indent=2)

# ---- summary.json ----
json.dump({
    "name": "My Eval",
    "generated_at": "2026-01-01T00:00:00Z",
    "counts": {"tasks": len(TASKS), "repositories": len(set(t[3] for t in TASKS)),
               "rollout_attempts": len(trial_rows), "trials": len(trial_rows)},
}, open(f"{OUT}/summary.json", "w"), indent=2)

# ---- release.json (local) ----
json.dump({
    "release_id": "local",
    "artifact_base_url": "artifacts",
    "artifact_patterns": {"trajectory": "trial-artifacts/{trial_name}/agent/trajectory.json"},
}, open(f"{OUT}/release.json", "w"), indent=2)

# ---- one task detail with an instruction ----
json.dump({
    "task_id": "fix-login-bug", "n_files": 2,
    "files": [
        {"path": "instruction.md", "language": "markdown", "size": 0,
         "content": "# Fix the broken login redirect\n\nAfter OAuth, users land on `/` instead of "
                    "their original page.\n\n- Preserve the `next` param through the redirect.\n- Add a test."},
        {"path": "task.toml", "language": "toml", "size": 0,
         "content": "[task]\nid = \"fix-login-bug\"\ntimeout_min = 90\n"},
    ],
}, open(f"{OUT}/tasks/fix-login-bug.json", "w"), indent=2)

print(f"Wrote {len(trial_rows)} trials, {len(tasks_rows)} tasks to ./{OUT}/")
print("Try: cp -r sample_artifacts/* artifacts/ && python3 -m http.server 8731")

