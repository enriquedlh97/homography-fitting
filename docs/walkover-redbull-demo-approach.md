# Walking-Over Red Bull Demo Approach

This note explains the current Red Bull replacement demo for
`data/melbourne-walking-over-logo.mov`. The goal is a presentation-ready
composite where the MELBOURNE court text, the left court logo, and the back-wall
logos are replaced while preserving the player's natural motion and avoiding
visible temporal jitter.

## Success Criteria

- Visual stability: no visible logo jitter across frames; measured corner jump
  and overlay acceleration stay at `0` for the current static baseline.
- Occlusion quality: the player should pass over the center court logo without
  the original MELBOURNE text leaking through in the visible foreground.
- Perspective quality: the two court-floor Red Bull logos should read as painted
  on the court plane, not as flat image stickers.
- Full composition: the demo should render all accepted placements together:
  center court Red Bull, left court Red Bull, and three black-wall Red Bulls.
- Runtime target: previews run on `H200` and `B200`, keeping the first successful
  result. The v68 two-second preview completed successfully on `H200`.

## Current Baseline

- Config: `configs/experiments/eval_walkover_v68_clicked_homography_static_preview.yaml`
- Output: `experiments/2026-04-30_16-22-17_walkover_v68_clicked_homography_static_preview_H200/outputs/composited.mp4`
- Crops: `experiments/2026-04-30_16-22-17_walkover_v68_clicked_homography_static_preview_H200/crops/`

v68 is the current presentation baseline because it combines the accepted v61
composition stack with improved court-plane perspective for the floor logos. v61
is still useful as the previous full-length reference, but v68 is the better
visual reference for explaining the homography direction.

## Pipeline Overview

The pipeline has five main stages:

1. Prompt and mask the original objects to replace.
2. Track each target region across the video.
3. Build a clean background where the original logos/text are removed.
4. Estimate or define a perspective-correct quadrilateral for each replacement.
5. Warp and composite the Red Bull artwork, using player occlusion masks where
   foreground motion crosses the replacement.

The current demo uses `video_hybrid` mode. This mode combines segmentation-based
tracking for the objects with explicit placement quads for the final composite.
That gives us a stable render path while still allowing the court placements to
come from a court-plane homography.

## Segmentation And Tracking

The project uses SAM-family segmentation to identify the regions that should be
replaced. In this run, the config uses `sam2_image` with SAM2. SAM2 receives
positive click prompts for each target object, returns object masks, and the
pipeline tracks those masks through the video.

The tracked masks are used for two different purposes:

- They define where the original content needs to be erased.
- They provide object support for fitting or applying the replacement geometry.

For the back-wall logos, this is enough because those surfaces are close to
fronto-parallel from the broadcast camera's point of view. The accepted wall
treatment uses tuned inpainting, feathering, and local blending to make the
black wall look continuous.

## Player Occlusion

The center court logo is harder because the player walks over it. If we simply
paint the Red Bull logo on top of the frame, it appears over the player's legs
and shoes, which breaks the illusion.

To solve this, the pipeline uses MatAnyone2 as a person matting model. At a high
level, MatAnyone2 estimates a soft alpha matte for the player in each frame. The
compositor uses that matte so foreground player pixels stay in front of the
replacement logo. This is why the center Red Bull can sit on the court while the
player still appears naturally above it.

The under-foot region required extra attention because motion blur and soft shoe
edges can preserve a small amount of original-frame color. Earlier iterations
tested clean-plate and targeted cleanup settings to reduce MELBOURNE text
survival while avoiding waxy or over-processed shoes.

## Clean Background Strategy

The court replacement depends on a clean plate: a version of the court where the
original MELBOURNE text is removed. The current stack uses a temporal median
clean video plus targeted cleanup parameters:

- Dilated and feathered clean-video blending to remove the source text.
- Luminance matching so the clean patch follows court lighting.
- Focused text cleanup around the MELBOURNE area.
- Under-foot decontamination to reduce text remnants near the player's shoe
  edge while preserving the natural foreground shape.

For the left court logo, the selected approach uses a compact erase/fill region
and a separate visible-logo quad. Separating the erase patch from the visible
logo avoids a large blue patch edge near the doubles line while keeping the Red
Bull artwork at the intended size.

## Homography And Perspective

A homography is a 3x3 projective transform that maps points from one plane to
another. In this demo, the useful mapping is:

```text
normalized court coordinates -> image pixel coordinates
```

Once that matrix is known, a rectangle on the court can be projected into the
camera view. This produces the correct four image coordinates for a logo that is
supposed to lie on the court surface.

The v68 baseline uses a fixed reference-frame court calibration:

1. We selected many points along the visible court boundaries.
2. The pipeline fit four court boundary lines from those samples.
3. Those lines were intersected into a clean court quadrilateral.
4. `cv2.findHomography` computed the mapping from normalized court space into
   image space.
5. The center and left Red Bull rectangles were projected through that mapping.
6. The projected quads were frozen into the v68 config for rendering.

This preserves the geometric benefit of a court-plane homography while keeping
the render temporally stable.

## Jitter Handling

The main jitter risk comes from estimating geometry independently on every frame.
Small line-detection changes from frame to frame can move the homography, which
causes the logo to shimmer even if the average perspective is plausible.

We tested dynamic court-plane placement in the v62-v64 sequence. That confirmed
the direction was useful, but it also showed that per-frame geometry estimates
can create visible instability on this clip.

The v68 baseline uses a reference-frame calibration instead. The homography is
computed once and the resulting logo coordinates are fixed for the preview. This
matches the demo requirement: the camera is stable enough over this short segment
that fixed calibrated quads produce better perceived quality than a noisy
per-frame estimate.

For a production version, the next step would be to combine reference-frame
calibration with temporal smoothing, confidence checks, and homography locking.
That would allow slow camera changes while still preventing frame-to-frame
jitter.

## Iteration History

- v53 established the earlier full-showcase walking-over checkpoint.
- v57 improved the black-wall Red Bulls, especially the left wall slot edge and
  vertical seam treatment.
- v58 improved the left court Red Bull erase strategy by using a compact court
  cleanup approach instead of a broad patch.
- v61 combined all accepted placements into one full-showcase render: center
  court, left court, and the three black-wall Red Bulls.
- v62-v64 explored dynamic court-plane projection for improved perspective, but
  the per-frame homography estimate introduced visible jitter.
- v65-v67 restored static stability and refined manual perspective, proving that
  stable placement was essential for the demo.
- v68 introduced fixed reference-frame court calibration, giving the left court
  logo much better floor-plane perspective without reintroducing jitter.

## What To Say In A Presentation

The key message is that the demo combines modern segmentation with classical
projective geometry:

- SAM2 finds and tracks the surfaces to replace.
- MatAnyone2 estimates the player foreground so the player can occlude the new
  logo naturally.
- Clean-plate compositing removes the original court text and wall content.
- A court-plane homography maps replacement artwork into the broadcast camera
  perspective.
- A fixed calibrated baseline prevents temporal jitter for the current demo.

This is a practical computer vision pipeline: learned models handle ambiguous
visual masks and people, while homography handles the deterministic geometry of
placing flat artwork on a known planar surface.

## Future Work

- Run the v68 baseline on the full clip once the short-preview perspective is
  accepted.
- Add a saved calibration artifact for the clicked court points and fitted
  homography so the derivation is fully reproducible.
- Use additional court landmarks and RANSAC for stronger calibration.
- Add temporal homography smoothing or lock/unlock logic for clips with stronger
  camera motion.
- Evaluate SAM3.1 for harder surfaces where SAM2 masks are less stable.
- Add automated visual regression crops for the left court logo, center logo,
  wall banners, and player foot-contact frames.
