# LADD HBB Learnability Audit Notes

- This audit is eval/no_grad and does not modify the checkpoint.
- Direct gap is d(q_s,u_t)-d(q_s,z_t); positive means z_t is closer to SAR q_s.
- Probe gap is R2(q_s->z_t)-R2(q_s->u_t); positive supports z_t being more SAR-learnable.
- High task_auc_u is not automatically a failure; u_t may be RGB-private but task-useful.
- Paired-vs-shuffled should be compared by running this tool twice with and without --shuffle-teacher-pairs.
- Gradient audit status: not_requested.
