# Court Floor Logo Replacement — Experiments Log

Test clip: `data/melbourne-walking-over-logo.mov` (player walks over MELBOURNE
text on court, 778 frames @ 59 fps).

Objective: replace MELBOURNE text with Red Bull logo, player walking over it
must look natural (no MELBOURNE leak-through, solid feet, no halos).

## Current Checkpoints (2026-04-30)

These are the source-of-truth outputs to use for the current walking-over
Red Bull demo and for future homography/perspective experiments.

### Full-showcase checkpoint: walking-over clip

**Clip:** `data/melbourne-walking-over-logo.mov`
**Baseline config:** `configs/experiments/eval_walkover_v68_clicked_homography_static_preview.yaml`
**Baseline output:** `experiments/2026-04-30_16-22-17_walkover_v68_clicked_homography_static_preview_H200/outputs/composited.mp4`
**Review crops:** `experiments/2026-04-30_16-22-17_walkover_v68_clicked_homography_static_preview_H200/crops/`
**Presentation notes:** `docs/walkover-redbull-demo-approach.md`

**Decision:** Treat `v68_clicked_homography_static_preview_H200` as the current
walking-over demo baseline. It keeps the accepted v61 composition stack while
upgrading the two court-floor logo placements to a fixed court-plane homography
calibrated from a reference frame. The result preserves temporal stability while
making the left-court Red Bull read much more naturally as a floor graphic.

**What changed since v61:** v61 remains the accepted full-length composite for
all five placements, but its two court-floor logos used manually supplied
screen-space quads. v68 fits a single court quadrilateral from user-selected
court-boundary samples, computes a fixed homography from normalized court space
to image space, and projects the court-logo rectangles through that matrix. This
keeps the no-jitter behavior of static placement while improving the perspective
of the floor logos.

**Validation:** The v68 H200 preview rendered 120 frames. Focused metrics passed
for jitter ratio, corner jump, logo area coefficient of variation, overlay
acceleration, and temporal SSIM. The evaluation script reports `inpaint_color_de`
as `N/A` for this setup, so the run exits nonzero even though the stability
metrics pass.

**Superseded outputs:**
- `experiments/2026-04-30_14-31-16_walkover_v61_all_redbulls_compact_court_full_H200/outputs/composited.mp4`
  remains the previous full-length showcase and the base composition stack for
  v68, but its court-logo perspective is less natural than the clicked
  homography baseline.
- `experiments/2026-04-29_16-32-37_walkover_v53_full_showcase_checkpoint/outputs/composited.mp4`
  was the previous full-showcase checkpoint.
- `experiments/2026-04-30_13-48-44_walkover_v59_v58_court_plus_v57_wall_full/outputs/composited.mp4`
  combined the accepted v58 court base with v57 wall banners, but the left-court
  patch still extended too far toward the doubles line.
- `experiments/2026-04-29_15-08-41_walkover_v52_checkpoint/outputs/composited.mp4`
  was only a checkpoint re-run of the accepted court-floor stack. It did not add
  the back black-banner treatment.

### Full-showcase checkpoint: tennis clip

**Clip:** `data/tennis-clip.mp4`
**Final config:** `configs/experiments/eval_original_v6_back_banner_checkpoint.yaml`
**Final output:** `experiments/2026-04-29_14-58-27_original_v6_back_banner_checkpoint/outputs/composited.mp4`
**Review crops:** `experiments/comparisons/original_v6_back_banner_checkpoint_crops/`

**Decision:** Treat `original_v6_back_banner_checkpoint` as the current
`tennis-clip` full-showcase checkpoint. It keeps the 7 accepted back banners and
the current court-floor targets, while removing the KIA side-panel target.

### Back-banner reference

**Reference clip:** `data/tennis-clip.mp4`
**Reference config:** `configs/experiments/eval_original_v5.yaml`
**Reference output:** `experiments/2026-04-25_20-28-35_original_v5_texture/outputs/composited.mp4`
**Crop evidence:** `experiments/comparisons/original_v5_back_banner_baseline_crops/`

**Decision:** `original_v5_texture` remains the accepted visual reference for
the back/background banner Red Bull logos. Use it to compare banner quality, but
use the two full-showcase checkpoint outputs above for the current deliverables.

**Next phase:** homography/perspective fixes for court logos, especially the
MELBOURNE-area Red Bull and the side-court Red Bull.

---

## Under-foot MELBOURNE Leak Investigation (2026-04-28)

**Diagnostic script:** `scripts/diagnose_underfoot_text_leak.py`
**Outputs:**
- `experiments/comparisons/underfoot_text_leak_diagnostics/`
- `experiments/comparisons/underfoot_text_leak_diagnostics_tight/`

**Symptom:** In playback, a small part of the original MELBOURNE word briefly
appears under/near the moving sole during foot contact, even though the broad
ghost-leg artifact is much improved.

**Evidence:** The diagnostic compares original, temporal-median clean plate, and
composite frames around frames 690-730. The `original-clean delta` images show
the MELBOURNE letters exactly where the original differs from the clean plate.
The `original survival` maps stay high in the moving shoe/sole band, and the
`suspected leak overlay` lights up under the shoe edge. v44, v46, and v49 all
show the same basic pattern, which explains why tuning broad text cleanup or
quad seams only gives marginal wins.

**Root-cause hypothesis:** This is foreground-edge contamination. The MatAnyone2
soft alpha / motion-blur band around the shoe contains original-frame RGB, and
the original RGB still includes MELBOURNE. Preserving that band keeps the shoe
natural, but it also preserves a little text under the sole. Replacing it too
aggressively removes the text but makes the shoe bright/waxy, which is what v45
showed.

**Next experiment direction:** Stop broad cleanup. Test edge-specific fixes:
1. underfoot decontamination that only targets original-like text pixels in the
   low-confidence sole band;
2. foreground edge extension / color decontamination that samples shoe-core color
   into the motion-blur edge while keeping the existing alpha.

---

## ⭐ CURRENT REVIEW CANDIDATE: v50 (2026-04-28, underfoot decontamination)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v50.yaml`
**Result:** `experiments/2026-04-28_20-44-25_walkover_v50_underfoot_decontaminate/`
**Review video:** `experiments/comparisons/v49_v50_v51_logo_crop_stacked.mp4`

**What it tried:** v49 plus `clean_video_underfoot_decontaminate`. This detects
pixels in the soft sole band that still behave like original-frame MELBOURNE
text by comparing original, clean, and current composite color vectors, then
blends only those pixels toward the clean plate.

**Outcome:** Best current review candidate. The underfoot diagnostic shows a
measurable reduction at the most obvious late contact frame: frame 730 suspected
leak pixels drop from 1603 (`v49`) to 1517 (`v50`), about 5.4%. The improvement
is not universal across every foot-strike frame, but blind QA preferred v50 over
v49/v51 twice, citing the best balance of under-foot remnant suppression and
natural shoe shape.

**Decision:** Promote v50 for user playback review. It is a targeted root-cause
fix, not broad cleanup, so it is less likely to repeat the v45 over-cleaning
failure.

---

## v51 (not promoted; 2026-04-28, shoe-edge color extension)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v51.yaml`
**Result:** `experiments/2026-04-28_20-45-31_walkover_v51_shoe_edge_extend/`

**What it tried:** v49 plus `clean_video_shoe_edge_extend`, sampling high-alpha
shoe-core color and extending it into contaminated soft-edge pixels.

**Outcome:** Not promoted yet. It is close to v50 and sometimes reduces suspected
leak pixels, but blind QA read it as slightly softer / more processed around the
shoe-logo interaction. Keep the idea, but tune more conservatively if revisited.

---

## v49 (superseded by v50; 2026-04-28, narrow text stripe)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v49.yaml`
**Result:** `experiments/2026-04-28_19-25-13_walkover_v49_narrow_text_stripe/`

**What it tried:** v44 plus a much narrower, gentler MELBOURNE-focus cleanup:
`clean_video_text_focus_box: [760, 880, 1180, 985]`,
`clean_video_text_focus_alpha_thresh: 0.62`, and
`clean_video_text_focus_replace_alpha: 0.85`. This keeps the cleanup away from
confident player pixels and avoids the broad replacement that hurt v45.

**Outcome:** Best current review candidate from the v48/v49 batch. Manual crop
inspection shows only subtle differences, but two blind QA passes both preferred
v49 over v44/v46/v48 for overall playback priorities: natural shoes first, then
seam/text artifacts. The remaining plume and rectangle are not fully solved.

**Decision:** Ask for user motion review of v49 against v44 and v46. Do not call
this final until playback confirms the shoe still feels natural.

---

## v48 (not promoted; 2026-04-28, temporal seam harmonization)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v48.yaml`
**Result:** `experiments/2026-04-28_19-27-28_walkover_v48_temporal_seam_harmonize/`

**What it tried:** v46 plus temporal smoothing of the quad-edge tone correction:
`clean_video_quad_harmonize_temporal_alpha: 0.25`. The idea was to reduce
frame-to-frame changes in the rectangular seam correction.

**Outcome:** Not promoted. It remains close to v46, but blind QA did not prefer it
over v49, and the crop differences are too small to justify making the seam branch
more complex yet.

---

## v42 (historical winner; 2026-04-28, temporal median + wider alpha cleanup)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v42.yaml`
**Result:** `experiments/2026-04-28_17-57-01_walkover_v42_median_alpha07/`

**Visual read:** Only a marginal improvement over v41. The wider
`clean_video_text_alpha_thresh: 0.7` does not materially remove the frame-730
blue/gray foot plume, but it also does not visibly damage the shoe or Red Bull
logo. Blind QA gave it a tie / tiny edge over v41.

**Interpretation:** The remaining plume is not solved by simply cleaning a wider
MatAnyone soft-alpha band. It likely needs a targeted post-blend plume cleanup or
a better player/motion-blur matte distinction, because the current bright-cleanup
condition mostly preserves the plume as if it were real shoe motion blur.

**Remaining:** Frame-730 foot plume and mild logo glow. v42 is the current best
by still-crop and mixed blind QA.

**User preference note (motion review):** User prefers `v44` for more natural shoe
appearance in playback. Until the next runs finish, treat `v42` and `v44` as
co-baselines (v42 = objective/crop baseline, v44 = motion-naturalness baseline).

---

## v43 (rejected; 2026-04-28, halo smooth 0.35)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v43.yaml`
**Result:** `experiments/2026-04-28_18-15-39_walkover_v43_median_halo_smooth035/`

**What it tried:** v42 plus `clean_video_halo_smooth: true` with
`clean_video_halo_alpha: 0.35`, replacing the low-confidence player halo zone
with the clean median plate before the main blend.

**Outcome:** Rejected by manual visual inspection. It expands the soft blue court
haze around the moving shoe at frames 700/730 and makes the foot plume more
noticeable. This confirms that broad halo replacement is too blunt for the
remaining artifact.

**Next:** Try reducing post-blend blur instead. The current 15px Gaussian blur may
be spreading the foot plume and vertical logo glow, so v44 should keep v42's
median clean plate and alpha threshold but lower the blur radius.

---

## v44 (inconclusive; 2026-04-28, reduced post-blend blur)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v44.yaml`
**Result:** `experiments/2026-04-28_18-28-54_walkover_v44_median_blur5/`

**What it tried:** v42 with `clean_video_post_blend_blur_px` reduced from 15 to 5,
testing whether the blur was spreading the frame-730 foot plume and vertical logo
glow.

**Outcome:** Inconclusive / not promoted. Manual inspection did not show a clear
foot-plume reduction. Blind QA split: one pass preferred v44 by a narrow margin,
the other preferred v42 because v44 looked broader/softer around the plume and
slightly less natural at the foot/logo collision.

**Decision:** Keep v42 as the current best because it is at least as good and has
the more established smoothing behavior. Do not keep tuning this single blur knob
unless motion playback shows a clearer benefit than the still crops.

---

## v45 (not promoted; 2026-04-28, text-focus cleanup)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v45.yaml`
**Result:** `experiments/2026-04-28_19-01-29_walkover_v45_text_focus_cleanup/`

**What it tried:** v44 plus a focused MELBOURNE cleanup box in the logo area:
`clean_video_text_focus_cleanup` with high-alpha replacement near the text zone.

**Outcome:** Not promoted. In blind 4-way comparison, v45 generally ranked lowest.
It did not produce a clear MELBOURNE leak win and tended to look slightly softer /
less natural at the shoe-logo interaction.

---

## v46 (candidate; 2026-04-28, quad-edge harmonization)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v46.yaml`
**Result:** `experiments/2026-04-28_19-00-07_walkover_v46_quad_harmonize/`

**What it tried:** v44 plus edge-only quad tone harmonization:
`clean_video_quad_harmonize` to reduce the visible rectangular boundary.

**Outcome:** Candidate. Blind 4-way comparison put v46 around 2nd/3rd depending on
reviewer weighting. It appears to help seam continuity without clearly harming shoe
naturalness, but gains are subtle and not decisive in stills.

---

## v47 (mixed; 2026-04-28, combined text-focus + harmonization)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v47.yaml`
**Result:** `experiments/2026-04-28_19-01-49_walkover_v47_text_and_harmonize/`

**What it tried:** combine v45 + v46 knobs in one run.

**Outcome:** Mixed. One blind reviewer ranked it best, another ranked it below
v44/v46 due to slight over-smoothing. Net: no decisive combined win yet.

**Current status:** Keep user-preferred `v44` as motion-naturalness baseline, with
`v46` as the most plausible seam-fix branch to inspect next in full playback.

---

## v41 (superseded by v42, marginally; 2026-04-28, temporal-median clean plate)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v41.yaml`
**Result:** `experiments/2026-04-28_17-40-30_walkover_v41_temporal_median_clean_plate/`

**Visual read:** Slight improvement over v40. The temporal-median clean plate
reduces the darker wrong-pose leg residue at frame 700 without bringing back
legible MELBOURNE text. The difference is subtle, but the artifact reads less
like a duplicated leg and more like a soft court/foot haze.

**What changed vs v40:** Instead of relying only on per-frame DiffuEraser clean
video plus threshold cleanup, v41 uses a derived clean video:
`data/clean_court_de_35px_temporal_median_quad.mp4`. Inside the MELBOURNE quad,
each pixel is replaced by its temporal median across the DiffuEraser clean video.
The goal is to keep stable court texture while rejecting moving player-shaped
residue from DiffuEraser's temporal propagation.

**QA:** Manual crop inspection and blind QA both give v41 a small edge over v40,
especially around frame 700. Frame 730 remains effectively tied.

**Remaining:** The main visible issue is now the soft blue/gray plume trailing the
moving foot around frame 730, plus mild vertical glow below the Red Bull mark.
Next iteration should target the foot plume / motion-blur halo without damaging
the real shoe or logo readability.

---

## v40 (superseded by v41; 2026-04-28, stronger clean-plate residue cleanup)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v40.yaml`
**Result:** `experiments/2026-04-28_17-16-24_walkover_v40_residue_cleanup_stronger/`

**Visual read:** Best result so far. The leg-shaped ghosts at wide-stance frames
500/700 are much less visible than v36/v38/v39. They are not fully gone, but the
artifact now reads more like a soft court haze/foot plume than a duplicated
wrong-pose leg.

**What changed vs v38:** Adds opt-in clean-plate residue cleanup before luminance
matching. The sanitizer detects local dark outliers inside the court quad in the
DiffuEraser clean plate and inpaints them, targeting the wrong-pose player smears
that DiffuEraser leaks through temporal propagation.

**v39 vs v40:** v39 was a conservative version and blind QA found it slightly
better than v38. v40 increases the detection radius/thresholds. Blind QA split
between v39 and v40, but manual crop inspection favors v40 because the visible
ghost legs are lower contrast at the key player-over-logo frames.

**Metrics:** Standard `quality_eval.py` metrics are unchanged from v38/v39:
jitter/pass, inpaint color/pass, temporal SSIM/pass; top-banner corner metrics
still fail due to the detector/metric not being meaningful for this court-logo
crop, not because of a visible court-floor regression.

**Remaining:** Soft haze under/near the moving feet and vertical glow below the
Red Bull mark. Next promising direction: static/semi-static court clean plate
from temporal median/texture reconstruction so the clean source contains no
video-inpainting player residue at all.

---

## v36 (superseded by v40; 2026-04-28, MatAnyone 2)

**Confirmed by blind QA vs v33 and v35.**

**Config:** `configs/experiments/eval_walkover_inpaint_first_v36.yaml`
**Result:** `experiments/2026-04-28_13-14-32_walkover_v36_matany2_full/`

**Key win over v33:** The sharp white triangular wedge artifact extending from
the right foot at frame 730 is GONE in v36. MatAnyone 2's improved fine-detail
boundary preservation eliminates this artifact.

Otherwise identical to v33's pipeline (DE PA 35px + lumin match + bright cleanup
+ post-blend blur), just swapping MatAnyone v1 → MatAnyone 2.

Remaining: leg-shape ghosts at wide-stance frames (540) — diagnosed as
DiffuEraser's per-frame player-position smearing leaking through. Fix in v37
(dark-pixel cleanup, running).

---

## v33 (superseded by v36)

**Confirmed by 3-way blind QA vs v26 and v34.**

**Config:** `configs/experiments/eval_walkover_inpaint_first_v33.yaml`
**Result:** `experiments/2026-04-28_06-44-50_walkover_v33_blur15/`

**Full config:**
```yaml
input:
  video: data/melbourne-walking-over-logo.mov
  clean_video: data/clean_court_de_35px.mp4  # DiffuEraser PA 35px
  clean_video_quad: [[649,840],[1268,840],[1268,1038],[649,1038]]
  clean_video_dilate_px: 25
  clean_video_core_dilate_px: 3
  clean_video_feather_px: 20
  clean_video_quad_feather_px: 60
  clean_video_temporal_window: 0
  clean_video_lumin_match: true
  clean_video_lumin_blur_px: 51
  clean_video_text_cleanup: true
  clean_video_text_threshold: 130
  clean_video_text_alpha_thresh: 0.7
  clean_video_post_blend_cleanup: true
  clean_video_post_blend_threshold: 130
  clean_video_post_blend_alpha: 0.3
  clean_video_post_blend_blur_px: 15
  logo: data/logos/redbull_white.png
pipeline:
  occlusion_masker:
    type: matanyone   # MatAnyone v1 continuous alpha
```

**Architecture summary:**
1. DiffuEraser player-aware (35px dilate) → clean court video, no MELBOURNE
2. MatAnyone v1 → continuous alpha for player matting
3. Lumin match (51px blur) → restore natural court shadow
4. Bright-pixel cleanup on original (alpha<0.7, gray>130) → kill MELBOURNE residue
5. Feathered blend → smooth player edges (20px) + quad boundary (60px)
6. Post-blend cleanup + 15px Gaussian blur on halo zone → final smoothing

**What v34 (`clean_video_full_inpaint`) tried but FAILED:**
- cv2.inpaint of entire halo zone introduced warm orange/red glow around player
- Over-aggressive cleanup loses player edge color fidelity
- Disqualified by blind QA

---

## v26 (superseded by v33)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v26.yaml`
**Result:** `experiments/2026-04-28_05-00-46_walkover_v26_clean_replace/`

**Architecture:** Inpaint-first + soft alpha + lumin match + per-pixel cleanup
1. **DiffuEraser player-aware clean video (35px dilate)** — text-removal-aware
   inpainter that DOESN'T hallucinate MELBOURNE text (ProPainter does!).
2. **MatAnyone v1** continuous alpha for player occlusion
3. **Luminance matching** restores natural shadow gradient on court
4. **Per-pixel bright-cleanup** — replace original bright pixels (alpha<0.5,
   inside quad, gray>130) with corresponding clean video pixel. Preserves
   lighting variation while killing MELBOURNE residue at motion-blur edges.

**Settings:**
- `clean_video: data/clean_court_de_35px.mp4` (DiffuEraser PA 35px)
- `clean_video_quad: [[649,840],[1268,840],[1268,1038],[649,1038]]`
- `clean_video_quad_feather_px: 60` (eliminates rectangle)
- `clean_video_feather_px: 20` (player edge smooth)
- `clean_video_lumin_match: true, blur_px: 51`
- `clean_video_text_cleanup: true, threshold: 130, alpha_thresh: 0.5`
- `occlusion_masker: matanyone` (continuous alpha)

**Remaining issues (acknowledged limits):**
- Subtle white halos at very-fast-motion frames (700, 730) — these are
  real camera motion blur artifacts that any composite would inherit
- Player slight color blend at extreme edges where MatAny alpha is mid-range

**Tried but didn't help / made worse:**
- v28 (halo_alpha 0.5 full replace): broke player colors with DE color shift
- v27 (alpha thresh 0.7): no improvement over v26
- v23 (threshold 130 alone): same as v26
- v15-v20 (PP-based clean): had MELBOURNE hallucinations
- ProPainter at any dilation: hallucinates MELBOURNE text from temporal context
- MatAnyone 2: works but doesn't fix the underlying clean video issues

---

## Earlier best: v12 (2026-04-28)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v12.yaml`
**Result:** `experiments/2026-04-28_02-12-39_walkover_v12_lumin_match/`

**Architecture:** Inpaint-first + soft alpha matting + luminance matching
1. ProPainter player-aware clean video (player preserved, court inpainted)
2. **Luminance matching:** at each frame, compute low-frequency Y ratio
   between original (with MELBOURNE white text suppressed) and clean inside
   the inpaint quad, apply to clean video. Restores the player's natural
   shadow on the court.
3. **MatAnyone v1** continuous alpha matting for occlusion
4. Feathered blend (player edge 20px + quad boundary 60px)

**Blind QA win over v10:** preserved natural shadow under/between legs at
frames 60, 100, 200, 530, 600, 700. Less "white halo" patches.

**New config keys:**
- `clean_video_lumin_match: true`
- `clean_video_lumin_blur_px: 51` (Gaussian kernel half-size)

---

## Previous best: v10 (superseded by v12)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v10.yaml`
**Result:** `experiments/2026-04-28_01-51-09_walkover_v10_propainter_matanyone/`

ProPainter PA + MatAnyone v1 — beat v7-real on halo, but still had subtle
"white halo" because inpaint removed the player's natural shadow.

---

## Earlier best: v7-real (superseded by v10)

**Config:** `configs/experiments/eval_walkover_inpaint_first_v7.yaml`
**Result:** `experiments/2026-04-28_01-24-17_walkover_v7_propainter_18px_REAL/`

**Architecture:** Inpaint-first
1. Pre-generate clean MELBOURNE-free video via ProPainter with **player-aware
   mask** (`mask = MELBOURNE_quad ∩ NOT (SAM2_player dilated 18px)`).
   Preserves player so model never tries to erase it.
2. Composite Red Bull logo over the clean court using SAM2 occlusion.

**Clean video generation** (`scripts/video_inpaint.py:run_propainter_player_aware`):
- ProPainter on H200, fp16
- Player mask: SAM2 large + Mask R-CNN auto-detection
- Dilation: 18px around SAM2 player (key for halo cleanup)
- Saves to `data/clean_court_propainter_player_aware/clean.mp4`

**Composite settings** (`src/banner_pipeline/pipeline.py:run_pipeline_video_hybrid`):
- `clean_video_quad`: `[[649,840],[1268,840],[1268,1038],[649,1038]]`
- `clean_video_dilate_px`: 25 (mask dilation for inpaint zone)
- `clean_video_core_dilate_px`: 3 (SAM mask dilation before feather)
- `clean_video_feather_px`: 20 (player edge feather radius)
- `clean_video_quad_feather_px`: 60 (quad boundary feather — eliminates rectangle)
- `clean_video_temporal_window`: 0 (not needed; clean video has player intact)

**Occlusion masker:**
- type: `sam2_video`
- checkpoint: `sam2.1_hiera_large.pt`
- mask_smooth: true, mask_close_px: 5, mask_dilate_px: 2

**Known minor issues:**
- Slight white shadow at bottom of player still visible (motion-blur edge)
- Faint dark shape between legs at some frames is mostly the player's racquet (real)

---

## Approach evolution

| Version | Date | Change | Result |
|---|---|---|---|
| Polished SAM2 overlay | 2026-04-27 | Pure overlay (no inpaint), court texture canvas, 25px feather | Baseline; small MELBOURNE text fragments at shoe soles |
| v1 inpaint-first | 2026-04-27 | DiffuEraser fast clean video + binary frame replace | Translucent feet, washed-out logo |
| v2 (heavy 25px dilate) | 2026-04-27 | Bigger SAM dilation in composite | Worst — banding + halo + still some ghosts |
| v3 feathered blend | 2026-04-27 | Feathered player edge alpha (no quad feather) | Better but visible rectangle |
| **v4 (quad feather)** | 2026-04-27 | Added 60px quad-boundary feather | Rectangle gone! Ghost legs visible |
| v5 temporal window | 2026-04-28 | Quick fix: union of SAM masks ±8 frames | Ghost legs partially hidden but still visible |
| v6 player-aware (5px) | 2026-04-28 | NEW: regenerate clean video with player-aware mask | Ghost legs gone but white halo visible |
| v7 player-aware (5px, hardcoded bug) | 2026-04-28 | Bug: dilation was 5 not 18 | Same as v6 |
| **v7-real (18px)** ⭐ | 2026-04-28 | Fixed bug; ProPainter 18px dilate | **Current best** |
| v8 DiffuEraser 18px | 2026-04-28 | Same architecture, DiffuEraser instead of ProPainter | More ghost-leg artifacts vs v7-real |
| v9 DE + MatAnyone | 2026-04-28 | DiffuEraser + MatAnyone soft alpha | Best halo cleanup, worst ghosts |
| v10 PP + MatAnyone v1 | 2026-04-28 | ProPainter + MatAnyone v1 | Beat v7-real on halo + ghosts |
| v11 PP + MatAnyone 2 | 2026-04-28 | ProPainter + MatAnyone 2 (CVPR 2026) | OOM at 16GB RAM, running with 64GB |
| **v12 PP + MatAny + lumin match** ⭐ | 2026-04-28 | Add per-frame luminance match | **Current best** — preserves shadow naturally |
| v13 PP 28px + lumin | 2026-04-28 | PP 28px dilate + lumin match | Better than v12 but trails still visible at frame 730 |
| v11 PP 18px + MatAny2 streaming | 2026-04-28 | Streaming MatAny2 inference (not process_video) → no OOM. Beats v12? | TBD blind QA |
| v15 PP 35px clean | 2026-04-28 | Bigger dilate to fully cover motion-blur trails | Done |
| v16 PP 28px + MatAny2 + lumin | 2026-04-28 | MatAny2 + 28px clean | Trails still visible at 730 |
| v17 PP 35px + MatAny + lumin | 2026-04-28 | 35px clean + MatAny v1 + lumin | Trail still visible (motion blur 200px+) |
| v18 PP 35px + MatAny2 + lumin | 2026-04-28 | All-in: 35px + MatAny 2 + lumin | Trail still visible |
| v19 PP 35px + temporal-window 4 (clean gen) | 2026-04-28 | UNION mask | **CRITICAL FINDING: ProPainter HALLUCINATES MELBOURNE text from temporal context, partial "MELBO..." visible in clean video** |
| v20 PP 35px+TW + MatAny + lumin (running) | 2026-04-28 | Composite using v19 clean | Running |
| DiffuEraser PA 35px ⭐ | 2026-04-28 | Switch to DiffuEraser, more aggressive text removal | **NO MELBOURNE TEXT in clean video at frame 730!** Color-shifts player but that's fine since composite uses original player |
| v21 DE35 + MatAny + lumin + cleanup | 2026-04-28 | DE 35px clean | Halos remain at player edges |
| v22 DE35 + clean both | 2026-04-28 | Cleaned original frame's bright pixels too | Failed (bug) |
| v23 DE35 + thresh 130 | 2026-04-28 | Lower bright threshold | Halos still visible |
| v24 DE35 + alpha thresh 0.3 | 2026-04-28 | Found bug: continuous alpha required threshold check | Marginal improvement |
| v25 DE35 + median replace | 2026-04-28 | Use median replacement instead of cv2.inpaint | Notable improvement, halos still subtle |
| v26 DE35 + clean pixel replace | 2026-04-28 | Replace bright original with corresponding clean pixel | Same as v25 (subtle improvement) |
| v27 DE35 + alpha 0.7 (running) | 2026-04-28 | More aggressive alpha threshold to catch more halo pixels | Running |

---

## Key insights

1. **Static MELBOURNE quad mask + ProPainter/DiffuEraser → ghost legs.** The
   model tries to erase the player AND the text, producing smeary partial-leg
   artifacts because of optical flow temporal context.
   **Fix:** per-frame mask = `MELBOURNE_quad ∩ NOT player_silhouette`.

2. **Inpainted court color differs from surrounding original court.** A hard
   quad boundary creates a visible rectangle.
   **Fix:** Feather the quad boundary 40-60 px inward.

3. **Motion blur halo around player = MELBOURNE white residue.** SAM2 binary
   mask can't represent partial-foot pixels. Result: white "shadow" / glow
   around feet when the composite blends original ↔ clean court.
   **Fix:** Dilate player mask 18+px in inpaint generation so halo gets cleaned.

4. **DiffuEraser produces smoother court but more residual leg artifacts**
   compared to ProPainter. ProPainter is more conservative / less hallucinatory.

5. **MatAnyone soft alpha helps with halo** but allows clean-video artifacts
   to peek through more at edges. Best paired with conservative ProPainter
   clean video, not DiffuEraser.

---

## Pending / next

- v10 (PP + MatAnyone v1): currently running
- v11 (PP + MatAnyone 2 / CVPR 2026): planned
- B200 ProPainter (full quality, original mask): still queued — full-res
  clean video would let us do a v12 at native 1912×1074 quality

## Files of interest

- Inpaint generation: `scripts/video_inpaint.py`
  - `_generate_player_aware_masks` — builds per-frame mask via SAM2
  - `run_propainter_player_aware` — full ProPainter player-aware run
  - `generate_player_aware_mask_video` — produces mask video for DiffuEraser
- Composite: `src/banner_pipeline/pipeline.py:run_pipeline_video_hybrid`
  - Clean video opening + per-frame replacement (search for "Clean video")
- Eval: `scripts/eval_court_experiment.sh` — runs `quality_eval.py` + extracts
  systematic crops at frames 0/50/100/200/300/500/700/730 (wide+tight+logo)
