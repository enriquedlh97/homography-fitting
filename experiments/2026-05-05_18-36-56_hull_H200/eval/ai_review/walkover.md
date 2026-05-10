# walkover — visual notes

This is the load-bearing test for the hypothesis: does feather=8 reduce the floor halo without breaking walkover realism?

Halo: yes, reduced. The diffuse bright glow on the matte court paint at the logo perimeter is meaningfully less obvious in entry (f685), pre-contact (f694), and exit (f723) than at feather=25.

Costs:
1. Crisp rectangular seam now traceable around the patch in the suspected-leak overlay and in entry/contact frames. The patch reads as stamped, not painted.
2. Player contact shadow remains poor — when feet land on the logo, no darkening propagates into the wordmark, so the player walks "above" the ad rather than across it.
3. Frame-to-frame jitter at the patch corners is visible across the consecutive_frames sheet — corners shimmer where the original Melbourne wordmark is rock-stable. Matches the +67% jitter ratio regression.
4. Occlusion of the wordmark by the player legs is acceptable but not convincing.

Net: halo improved at the cost of seam + jitter visibility. Qualitatively a lateral move, not a clear win.
