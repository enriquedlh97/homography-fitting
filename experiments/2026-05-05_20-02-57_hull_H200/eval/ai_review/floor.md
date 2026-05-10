# floor

`erase_text=true` does what it advertises: the faint MELBOURNE letter ghost that survived around the Red Bull mark in P3-A5 is gone, and the court surface around the wordmark now reads as a clean uniform blue paint. The Red Bull mark itself is unchanged — crisp, no halo, no letter-edge reflex, no jitter, no perimeter seam.

The trade-off is that the inpainted region is slightly smoother than the surrounding matte court paint. A scrubbing viewer who knows where to look will see a region of the floor that lacks the natural micro-grain texture of the rest of the court. It does not jump out at normal playback speed, but it costs a point on `texture_match` and a point on `painted_on_vs_pasted_on`.

This is exactly where the floor SSIM regression (0.975 -> 0.899) is coming from — the metric is being fooled because we removed real texture (the MELBOURNE letters) that it was previously matching.
