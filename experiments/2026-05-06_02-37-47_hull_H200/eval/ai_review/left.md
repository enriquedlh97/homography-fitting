# left — inpaint_method=ns variant (a2)

Left diagonal banner with Navier-Stokes inpainting in place of Telea. The targeted dimension `left.edge_reflex` does not move from 4: NS produces a residual letter-edge softness that's visually indistinguishable from Telea at strip resolution. The bottleneck is downstream of the inpaint method. No new halo; geometry and color stable. Left is essentially equivalent to baseline a2.
