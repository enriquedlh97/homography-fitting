# walkover — visual notes

Across f685 (entry) → f723 (exit), the player walks over the court-floor Red Bull logo. Tracking is solid: in the consecutive_frames strip the logo holds its position frame-to-frame as feet pass through, with low jitter and no obvious tearing. The forensic sheets (delta, survival, suspected-leak) confirm the logo footprint is being preserved through occlusion.

What breaks the painted-on illusion in this window is not motion — it's the floor halo (still present, since this run did not touch floor params) and the absence of a contact shadow where the foot meets the logo. The leak overlay shows residual color seepage at the bull/wordmark perimeter, consistent with the same halo seen in the floor crops_strip. Painted-on score is held to 3 by these surface-realism issues, not by walkover-specific failure.
