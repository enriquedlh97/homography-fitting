# left — visual notes

The original YoPRO ad on the diagonal side banner is a clean white wordmark with no edge artifact — that is the quality bar.

Our Red Bull composite shows the canonical rubric-v2 reflex/smearing artifact: a faint horizontal ghost or duplicate along the top and bottom edges of "RedBull" letters, and the rings around the bulls show subtle paint drift. It is not loud, but it is present on every frame, and a viewer scrubbing the clip would catch it because the original is so clean. Score 2/5 on edge_reflex per the calibration callout — not 5.

Halo is not a problem here; the matte blue banner hides any feather glow.

This region is not a target of the floor alpha_feather change, so behavior is essentially unchanged versus the feather=25 baseline. The reflex artifact lives in the warp/sampling chain, not the compositor feather, so it cannot be addressed by this hypothesis.
