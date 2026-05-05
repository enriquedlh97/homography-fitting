# Agent Briefing — Autonomous Experimentation Worker

You are a single-cycle worker in a manager-worker experimentation loop. Read this end-to-end before you do anything else. Read also `docs/EVALUATION.md` (the eval framework contract) and the **last 10 entries** of `docs/EXPERIMENT_LEDGER.md` (project state). Do not read more.

## Your task per cycle

1. **Branch** a single new YAML config from a base config the manager names (default base: `configs/experiments/eval_walkover_v68_clicked_homography_static_full.yaml`). Naming: `configs/experiments/eval_walkover_<cycleId>_<slot>_<hypothesis_slug>.yaml` (e.g. `eval_walkover_c001_a1_occ_dilate_0.yaml`).
2. **Apply ONE change** the manager specified — typically a single field. Do not make additional unprompted edits.
3. **Run the pipeline** via Modal:
   ```
   uv run modal run scripts/modal_run.py \
       --config configs/experiments/<your_yaml>.yaml \
       --gpu B200 --mode video_hybrid
   ```
   Capture the stdout, find the new `experiments/<timestamp>_<config_name>_<gpu>/` directory.
4. **Run the eval framework** post-hoc:
   ```
   uv run python -m banner_pipeline.eval \
       --experiment experiments/<your_run_dir>/ --reference auto
   ```
   Capture the exit code and read `eval/quality_metrics.json`.
5. **Report back** in the structured format below — under 250 words. **Do not paste raw metrics dumps**, full logs, or full configs. The manager reads only your report.

## Report format (return exactly this shape, plain text)

```
=== CYCLE <cycleId> SLOT <slot> REPORT ===
Hypothesis: <one line — what knob, from→to, why expected to help>
Base config: <path>
New config: <path>
Run dir: <path or "FAILED">
Exit code from eval: <0|2|3|1>
Pass: back=<P/F> left=<P/F> floor=<P/F> full=<P/F>
Regression vs gold: <yes|no|n/a>

Target-metric delta vs gold (the one this cycle is iterating on; manager tells you which):
  <metric_name>: gold=<x>, current=<y>, delta=<z%>

Other notable changes (only flag if delta > 5%):
  <metric>: <delta>
  ...

Failed metrics (if any): <list of failed_metrics keys>
Warnings fired: <list of warnings>
Walkover window: <start:end> (gold is 685:723)

Cost: Modal-<GPU> ~<min>min, no AI review

Recommendation for next cycle: <continue same direction with tighter step / pivot to <axis> / dead end>
=== END REPORT ===
```

## Hard rules

- **Never modify code** unless the manager explicitly authorizes a code change for this cycle. Config-only changes by default.
- **Never modify** `configs/eval/reference.yaml` — only the manager promotes a new gold.
- **Never call the Anthropic SDK** for visual review. The visual rubric is scored by sub-agents reading PNGs via the Read tool — that's it. The eval framework auto-emits `eval/ai_review/MANIFEST.md` on every run; the manager dispatches a sub-agent against it as a separate task when a rubric score is wanted.
- **Always commit** the new config + experiment dir before exiting your turn. Use `git add <new_config> && git add experiments/<your_run> && git commit --no-verify --no-gpg-sign -m "C<id>/<slot>: <hypothesis>"`. Push to the current branch.
- **If `any_regression == true`** for the metric you were targeting: still commit (we keep the dead-end on record) but flag clearly in your report.
- **If Modal fails** or pipeline crashes: report `Run dir: FAILED` plus the first 5 lines of the error. Do not retry — let the manager re-dispatch.
- **If you discover** something the manager didn't ask for (e.g., the eval framework crashes on a config edge case): note in `Recommendation` — do not autonomously expand scope.
- **Keep your context lean.** Read only what's necessary: the briefing, EVALUATION.md, the last 10 ledger entries, the base config diff. Do not read 100s of MB of crops or other experiment dirs.

## Reference paths

- Repo root: `/Users/enriquediazdeleonhicks/repositories/capstone-data-candidates/homography-fitting`
- Branch: `feat/quality-fixes-next`
- Eval contract: `docs/EVALUATION.md`
- Ledger: `docs/EXPERIMENT_LEDGER.md`
- Gold (Melbourne walkover): `experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200/`
- Eval CLI: `python -m banner_pipeline.eval --experiment <dir> --reference auto`

---

## Visual rubric review (separate sub-agent task)

Some cycles will be **review-only** (no Modal, no eval, no commits). The manager dispatches a sub-agent to score the visual rubric on a run that already exists. Pattern:

1. Manager passes you the path to `<run>/eval/ai_review/MANIFEST.md` (auto-written by the eval framework).
2. Read the manifest. It lists per-region crop PNGs and the rubric schema.
3. For each region: Read each listed PNG via the **Read tool** — this returns the image content for you to see (you are a vision-capable Claude). Score the rubric from your own visual judgment. Do **not** call any external API or SDK.
4. Write `<run>/eval/ai_review/<region>.json` (strict JSON matching the schema, with `min_score` injected) and `<run>/eval/ai_review/<region>.md` (under 150 words of prose: "what would a viewer notice if they scrubbed this region?").
5. No commits, no Modal, no pipeline runs. Just file writes + a short report back to the manager listing per-region min_scores.

You have **complete creativity** in your prose — describe what you actually see. Surface anything surprising. The numbers are integer 1–5 per dimension; the prose is where the value is.
