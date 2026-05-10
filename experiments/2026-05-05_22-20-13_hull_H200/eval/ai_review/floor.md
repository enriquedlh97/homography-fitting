## Floor — visual review

The floor strips (early/mid/late) cover frames 110-656 — none of them include the walkover window where players cross the mark, so they look the same as P3-A17: clean Red Bull wordmark on uniform blue court, no halo, no reflex, no jitter. Crops_strip f0000-f0766 confirms identical-looking placement throughout the clip.

The new shadow_strength=0.5 synthesis only fires where the MatAnyone player mask intersects the floor logo; on these no-player strips it is a no-op, which is the intended default. See walkover.md for the actual contact-shadow assessment — that's where the new code path is exercised.

Texture stays at 3 (inpainted plate slightly smoother than matte court micro-grain) and painted_on_vs_pasted_on at 4 (same as P3-A17 baseline; the plate reads as a slightly cleaner-than-court rectangle if you look closely, but the absence of MELBOURNE bleed-through keeps it well above pasted-on). All other dimensions: 4-5.
