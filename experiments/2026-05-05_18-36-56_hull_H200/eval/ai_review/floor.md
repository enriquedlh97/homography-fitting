# floor — visual notes

A scrubbing viewer would notice that the soft luminance glow that previously surrounded the Red Bull floor logo has receded — the diffuse bright halo on the matte blue court paint is genuinely diminished versus the feather=25 baseline (P2-C012/A2). That is the win this hypothesis was chasing.

The cost shows up immediately, though: the patch now reads as a rectangle stamped onto the court rather than a glow blended into it. In the walkover forensic sheets and the suspected-leak overlay, you can trace a faint linear step around the patch perimeter that was not visible in the original baked-in Melbourne wordmark above. Letter reflex on "Red Bull" wordmark looks about the same as the baseline (subtle ringing on the curves of B and R), so the feather change did not help that axis.

Most concerning is temporal: across motion strips, the patch boundary shimmers frame-to-frame. This matches the +67% jitter regression — feather=25 was masking the H estimator's per-frame instability by smearing the boundary; feather=8 exposes it. Halo improved, but seam visibility and jitter both degraded. Net qualitatively: a wash, possibly worse for broadcast realism.
