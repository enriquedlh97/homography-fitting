## Walkover — visual review

**Headline finding:** `temporal.player_contact_shadow` moves 4 → 5 vs P3-A17 baseline. The new shadow synthesis is doing what was intended.

In P3-A17 the Red Bull mark stayed flat-bright everywhere except the binary foot-cutout — the player's shoes read like they were hovering above a sticker. The consecutive_frames strip across f0685-f0724 in P3-A28 shows the wordmark and the bulls visibly dimming in a soft halo around each foot. The dimming is most prominent at f0700-f0708 where the shoe sole transits the "Red Bull" lettering: the white text loses some of its peak brightness in a feathered region 10-20 px around the foot, then fades back to full brightness further out.

Forensic_sheet_contact_f0704 composite confirms the falloff is smooth (Gaussian-blurred dilation of the player mask, not a hard ring). The effect reads as a soft cast shadow grazing the floor — photographically plausible.

No halo, no reflex, no jitter introduced. The synthesis is a per-frame multiplicative darken of the inserted logo pixels — does not change the patch boundary, so all the P3-A5 stability gains are preserved.

The remaining 4 on `painted_on_vs_pasted_on` is the inpainted-plate smoothness (orthogonal to this axis), not a contact-shadow artifact.
