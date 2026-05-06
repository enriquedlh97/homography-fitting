# floor — visual review

The floor logo is where this experiment's tradeoff lives. The very tight alpha_feather=2 successfully eliminates the halo glow that v1 reviewers flagged on earlier runs — halo_presence scores a clean 5, and a scrubbing viewer would not see any luminance bloom around the Red Bull wordmark or bulls icon. That is the win.

The cost is edge_seam_visibility. With feather=2 + mask_dilate=8, the patch has a thin but perceptible rectangular perimeter where the Red Bull bake meets the surrounding court paint. BTN's stable per-frame H prevents the seam from walking frame-to-frame (so it does not pop like P3-A4/a2's line-based jittery H did), but the seam is still locatable on slow scrub, especially along the top edge in mid/late strips.

Texture match to the matte court is good. Geometry is solid. Net: this variant trades P3-A5's clean blend for halo elimination — overall slightly worse.
