# walkover — short prose

The walkover (frames 685-723) is the critical evaluation surface. Versus
P3-A12 (erase_text alone) and the P3-A5 baseline:

- Bleed-through of the original MELBOURNE wordmark under the player's feet
  is eliminated. The suspected-leak-overlay panel on the contact f0704
  forensic sheet shows a clean Red Bull mark with no ghost characters from
  the underlying paint. consecutive_frames over the full 21-frame window
  shows clean limb cutouts with no per-frame leak.
- Halo and edge reflex on the small floor mark: absent. The rubric-v2
  calibration artifacts do not appear.
- player_contact_shadow stays at 4 (NOT 5): at f0704 the cast shadow under
  the planted feet is plausible, but at post-contact f0713 there is a
  darker streak/trail above the wordmark that reads more as a dilation-
  buffer-induced ghost than as a true instantaneous cast shadow. Combining
  occlusion_dilate_px=8 with erase_text did NOT push contact_shadow from 4
  to 5 — the artifact is the dilate buffer itself.
- Texture and occlusion both at 4 not 5 because the dilation creates a
  thin halo of clean court paint around the player.

Net: this is a clear win over the baseline; we are now bottlenecked on
texture and contact-shadow synthesis, not bleed-through.
