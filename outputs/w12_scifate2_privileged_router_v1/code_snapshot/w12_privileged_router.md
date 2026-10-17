# W12: privileged dynamics during training, S-only structural routing at deployment

Fixed follow-up, 2026-10-17, after W10 source-only frozen interventions and W11
cohort retraining. Frozen real-dynamics experts retain most of their advantage
under averaged routing, but training a cohort-only router does not improve over
its stronger cohort-static control. Test whether dynamics helps learn expert
responsibilities during training rather than being essential for cell-wise
routing at deployment. This is an exploratory mechanism test, not a lineage claim.

Use the direct-distance W10 b1 teachers, 12000 steps, ALL THREE teacher arms and
all three seeds. Freeze each teacher's state backbone and experts. Warm-start a
same-size student router from that teacher; mask its velocity channel to zero.
Train router only on source S and source-side teacher soft responsibilities from
r1–r3, fixed 4000 steps, batch512, AdamW1e-3, wd1e-4. No future expression enters
student distillation, no r4 optimizer input, no early stopping or grid. For the
shuffled teacher, reproduce its exact per-cohort velocity permutation.

Nine lightweight fits, one final checkpoint each. Report each matched teacher
family, student-vs-teacher source KL, full predictive energy/variance/covariance,
and all three windows using the same frozen evaluation rows. Arm names refer to
the teacher's training privilege, NOT student inference inputs. In every family
the local student is S-only, while the shared frozen MEAN predictor still uses
analytic population dynamics. Do not claim the whole deployed model is U-free.

Compare the real-dynamics-trained student against BOTH static-trained and
shuffled-trained students, the original b1 real teacher, and W11's stronger
cohort-static reference. Report failure if the student cannot retain the benefit.
Do not change the W10 bridge or automatic long continuation. GPUs1–3 only; GPU0
and GPUs4–7 untouched.
