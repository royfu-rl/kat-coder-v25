You are critiquing one completed Harbor task trial. Explain whether the trial
is a useful evaluation signal, attribute the outcome carefully, and ground
claims in artifacts. Do not solve the task, modify files, or rerun the verifier.

## Inspect
1. `{trial_dir}/result.json` — recorded outcome, reward, run errors
2. `{trial_dir}/verifier/`, `/artifacts/`, `/agent/` — verifier output, transcript
3. `{task_dir}/instruction.md`, `task.toml`, relevant task files

Some tasks include hidden tests, reference solutions, setup scripts, rubrics.
These are context, not automatic proof the solver should have known something.
If a requirement is only clear from hidden artifacts, treat that as task-side
unfairness unless public materials made it reasonably inferable.

## Judge
- Decide what happened and whether the trial is a fair signal.
- Every trial falls into one of four quadrants — **TP** (real pass), **FP**
  (cheated pass), **TN** (real failure attributable to the solver), **FN**
  (failure not attributable to the solver). Pick exactly one `tag`; when the
  tag is a `TN` sub-tag, it also identifies which stage of the SWE pipeline
  the failure occurred at.
- Separate solver mistakes from task ambiguity, broken verification, missing
  artifacts, infrastructure failures, or shortened budgets.
- For passing trials, check whether the pass appears legitimate. Only call
  cheating when artifacts support it.
- Use reference solutions to understand intent, but do not require exact match.

## Evidence
Back claims with `path:line` + short quote. Don't fabricate. Missing/
contradictory evidence → lower confidence, don't guess.

## Output (single JSON to {critique_result_path})

Write a single valid JSON object to `{critique_result_path}`. Optional supporting
artifacts may go under `{critique_artifacts_dir}`. Your final response must be
only the same raw JSON, with no markdown fences or commentary.

```
{
  "tag":  // TP
          "PASS_LEGITIMATE"
          // FP
        | "PASS_CHEATED"
          // TN — 1. Localization
        | "LOCALIZATION_WRONG_FILE" | "LOCALIZATION_WRONG_FUNCTION"
        | "TASK_MISINTERPRETED"
          // TN — 2. Patch generation
        | "PATCH_PARTIAL_REQUIREMENT" | "PATCH_MISSING_BRANCH"
        | "PATCH_LOGIC_ERROR"
          // TN — 3. Patch application
        | "APPLY_MALFORMED_PATCH"
          // TN — 4. Runtime integrity
        | "RUNTIME_UNRESOLVED_REFERENCE" | "RUNTIME_STALE_CALLSITE"
          // TN — 5. Fail-to-Pass
        | "F2P_UNCOVERED" | "F2P_NO_PATCH_EMITTED" | "F2P_TESTS_NOT_RUN"
          // TN — 6. Pass-to-Pass
        | "P2P_SIGNATURE_BREAK" | "P2P_IMPORT_TIME_BREAK"
          // TN — 7. Trajectory-level
        | "TRAJECTORY_INCOMPLETE"
          // FN
        | "TEST_MISMATCH" | "TEST_BROKEN"
        | "AMBIGUOUS_TASK" | "EXECUTION_EXTERNAL"
          // fallback
        | "UNCLEAR",
  "quadrant": "TP" | "FP" | "TN" | "FN" | "UNCLEAR",
  "feedback": string,           // 3-5 sentences
  "trial_outcome": "pass" | "fail" | "incomplete" | "unclear",
  "evidence": string[],         // 2-6 load-bearing citations
  "confidence": "low" | "medium" | "high"
}
```

Field notes:

- `tag`: pick exactly one — see `Tag definitions` below.
- `quadrant`: the four-way summary — must be consistent with `tag`.
- `feedback`: 3–5 sentences covering what happened, attribution, and any
  fairness concerns.
- `evidence`: 2–6 load-bearing citations in `path:line` form (short quote or
  fact after each).
- `confidence`: use `low` when key artifacts are absent, incomplete, or
  contradictory; low-confidence FP/FN judgments should fall back to `UNCLEAR`.

### Tag definitions

Pick the single tag whose *root cause* best matches. When two tags could apply,
choose the one closest to the underlying cause (not the surface symptom) and
mention the secondary factor in `feedback`. Within TN, prefer the **earliest**
pipeline stage that can independently explain the failure — early defects are
root causes, later symptoms are cascades. When evidence is insufficient to
disambiguate at medium-or-higher confidence, prefer `UNCLEAR` over guessing.

#### TP — real pass

- **`PASS_LEGITIMATE`** — Verifier passed and the patch genuinely satisfies
  the task. No sign of shortcut, hardcoding, or leaked answers.
  *Counter-example:* the patch only makes tests pass by editing the tests
  themselves → `PASS_CHEATED`.

#### FP — cheated pass

- **`PASS_CHEATED`** — Verifier passed, but artifacts show the pass is not a
  real solution. Includes: modifying or deleting hidden tests, hardcoding
  expected outputs, fetching the upstream fix via network or `git log`,
  `npm install`-ing / `pip install`-ing a post-fix version, or otherwise
  short-circuiting the intended work. **Requires direct artifact evidence** —
  do not infer from "the fix looks suspiciously clean".
  *Example:* transcript shows `curl https://github.com/.../pull/1234.diff`
  followed by applying that diff verbatim.

#### TN — real failure, solver's responsibility (organized by SWE pipeline stage)

**1. Localization** — did the solver find the right place to change?

- **`LOCALIZATION_WRONG_FILE`** — Modified files are disjoint from the gold
  patch's file set.
- **`LOCALIZATION_WRONG_FUNCTION`** — Right file, wrong function/method/class.
- **`TASK_MISINTERPRETED`** — Intent inferred from the issue contradicts the
  gold patch's intent (e.g. solver thought this was a rename, task wanted a
  behavior change).

**2. Patch generation** — given the right location, was the patch itself right?

- **`PATCH_PARTIAL_REQUIREMENT`** — Multi-part task; implemented some
  sub-requirements, dropped others.
- **`PATCH_MISSING_BRANCH`** — Handled some branches / parameters / edge cases
  within a single logic path, missed the rest.
- **`PATCH_LOGIC_ERROR`** — Right location, right direction, but the core
  logic is wrong or too shallow.
  *Example (Boa cancellation):* solver built the cancellation graph but forgot
  to hook `Promise.then` callbacks into it.

**3. Patch application** — does the patch text actually apply cleanly?

- **`APPLY_MALFORMED_PATCH`** — Patch is syntactically incomplete, truncated
  mid-hunk, or otherwise fails `git apply`. The patch never made it into
  the repo.

**4. Runtime integrity** — patch applied, but did the code even import?

- **`RUNTIME_UNRESOLVED_REFERENCE`** — Code references an undefined or
  unimported symbol; test collection or import raises `NameError` /
  `ImportError` / `AttributeError` before any test logic runs.
- **`RUNTIME_STALE_CALLSITE`** — Solver changed a signature but didn't update
  all callsites; `TypeError: missing required argument` at runtime.

**5. Fail-to-Pass** — did the F2P hidden tests actually exercise the fix?

- **`F2P_UNCOVERED`** — Tests ran and passed, but never exercised the F2P
  target scenario (the fix happens to be untested).
- **`F2P_NO_PATCH_EMITTED`** — Trajectory produced no real patch to evaluate.
- **`F2P_TESTS_NOT_RUN`** — Verifier never ran the hidden tests at all.

**6. Pass-to-Pass** — did the patch break something previously green?

- **`P2P_SIGNATURE_BREAK`** — Solver changed a shared API signature and broke
  existing callers elsewhere in the repo.
- **`P2P_IMPORT_TIME_BREAK`** — Import-time or initialization-time side effect
  regressed unrelated modules.

**7. Trajectory-level** — the run itself didn't complete for solver reasons.

- **`TRAJECTORY_INCOMPLETE`** — Budget cut short mid-work, process killed,
  transcript ends before a submission. The solver ran out of runway.
  *Counter-example:* API returned 5xx → `EXECUTION_EXTERNAL` (FN).

#### FN — failure not attributable to the solver

- **`TEST_MISMATCH`** — Hidden test asserts implementation details of the
  gold patch (white-box); behaviorally equivalent solutions get rejected.
  *Example:* test asserts a specific variable name in the traceback.
- **`TEST_BROKEN`** — Verifier itself has a bug: test references a nonexistent
  file, rubric logic is wrong, grader crashes on a valid solution.
- **`AMBIGUOUS_TASK`** — Requirement only inferable from hidden artifacts, or
  `instruction.md` is self-contradictory. Public materials do not make the
  answer reasonably derivable.
- **`EXECUTION_EXTERNAL`** — Infrastructure failure outside the solver's
  control: API 5xx, sandbox OOM (not caused by solver commands), network
  flake, model provider rate-limited.

#### Fallback

- **`UNCLEAR`** — Evidence is missing or contradictory enough that none of the
  above can be assigned with medium-or-higher confidence. Pair with
  `"confidence": "low"` and cite what's missing in `evidence`.

---

## Why these FN categories?

The four `FN` tags mirror the failure modes that
[SWE-bench Verified (Chowdhury et al., 2024)](https://openai.com/index/introducing-swe-bench-verified/)
identified when a team of software engineers hand-audited the original
[SWE-bench (Jimenez et al., 2023)](https://arxiv.org/abs/2310.06770). Their
audit found that a non-trivial fraction of "failed" trials in SWE-bench were
not the model's fault at all — they collapsed into three families, which we
lift almost verbatim:

- Task descriptions were **underspecified**, making the correct behavior
  impossible to derive from the public prompt alone — our `AMBIGUOUS_TASK`.
- Hidden tests were **overly specific**, rejecting valid solutions that
  differed from the gold patch in incidental ways — our `TEST_MISMATCH` (for
  white-box coupling) and `TEST_BROKEN` (for outright buggy verifiers).
- **Environment / harness issues** during grading — our `EXECUTION_EXTERNAL`.

SWE-bench Verified addressed these by producing a hand-cleaned subset (500
samples) where verdicts are trustworthy. We can't afford dataset-level human
review on every rollout, so `FN` is that same taxonomy operationalized at the
trial level: whenever the judge determines a failure belongs to one of these
buckets, the trial's reward is filtered out of the training signal
downstream. The categories are chosen to be exhaustive for the "not-the-
solver's-fault" quadrant precisely because Verified already demonstrated that
a smaller set doesn't cover the real distribution of noise.

TN, in contrast, is organized by *where in the SWE pipeline the defect
occurred* (localization → generation → application → runtime → F2P → P2P →
trajectory), following the pipeline view established by SWE-bench and
[Agentless (Xia et al., 2024)](https://arxiv.org/abs/2407.01489). The two
dimensions are orthogonal: FN answers "does this trial count?", TN answers
"if it counts, where did the solver go wrong?".

