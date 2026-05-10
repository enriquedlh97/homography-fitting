# walkover

This is the region the experiment was chasing. Compared frame-for-frame against the P3-A5 baseline:

- **Contact shadow improved** (P3-A5 = 3 -> P3-A12 = 4). The MELBOURNE letter ghost that was bleeding through under the player feet in P3-A5 is substantially reduced. The suspected-leak-overlay panel still shows residual red flecks where the original M-L-B-O-U-R survival hits the player-occluded area, but in the actual composite panel the Red Bull mark stays consistently clean under the legs across entry/pre-contact/contact/post-contact/exit. P3-A8's prediction was correct.
- **Texture match drops** (4 -> 3). Same inpaint-too-smooth artifact as the floor region. A scrubbing viewer would see the patch read slightly cleaner than the surrounding court.
- **No new artifacts.** Consecutive-frames panel is stable across f0685-f0724; no jitter, no halo, no edge reflex, no rectangular seam. The faint dark plume at the top of the wordmark on post-contact f0713 reads as a player shadow rather than a synthesis defect.

Net: walkover gets the win. Floor takes a small texture hit. The walkover_iou regression (0.985 -> 0.788) is almost certainly the metric reacting to the inpainted region's smoother texture rather than to a visible quality drop.
