# floor — visual notes

Floor region is unchanged in this sweep (banner-only `mask_dilate_px` override), and the visuals confirm it: the Red Bull floor logo on the matte Melbourne court still shows a visible bright halo around the wordmark and the bull silhouettes. The original (top row) is the painted Melbourne wordmark — flat, matte, no glow — which exposes the composite's halo clearly. Letter-edge reflex is faint but present.

Geometry, hue, saturation, and size are all good; the artifact family here is purely the halo + slight texture mismatch (the composite reads slightly glossier than the matte court paint). Occlusion handling around the player's feet looks adequate but lacks contact shadow, which keeps the logo reading as overlaid rather than painted. No regression versus baseline expected or seen.
