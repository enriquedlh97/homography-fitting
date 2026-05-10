# Floor logo — review

The Red Bull floor mark reads as painted-on across the full motion strip. Court tone, lane lines and the back of the court all show through beneath the wordmark consistently with how the original Melbourne wordmark sits.

**Gate-active vs gate-dormant comparison:** I A/B'd the late motion strip (frames 626–656) against the P2-C010/A2 gate-dormant run. They are visually indistinguishable at the floor region. The 16 hybrid_lock ramping frames did NOT introduce wobble, jitter or color shimmer here — the small `floor_roi_jitter_ratio` regression in the metrics does not surface to the eye. Conversely, no clear motion-tracking improvement is visible at this resolution either; both variants already nail the static-camera assumption on the floor.

Net: the gate fires safely on this run. Floor region is broadcast-grade in both variants.
