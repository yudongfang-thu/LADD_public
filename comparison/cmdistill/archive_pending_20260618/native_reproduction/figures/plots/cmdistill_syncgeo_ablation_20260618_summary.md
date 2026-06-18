| run | rows | last epoch | latest mAP50 | best mAP50 | best epoch | latest mAP50-95 | best mAP50-95 | best mAP50-95 epoch | latest KD | latest feat | latest rel | latest out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sync geo no KD | 300 | 299 | 0.6410 | 0.6853 | 173 | 0.3588 | 0.3871 | 222 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sync geo feat+rel KD | 300 | 299 | 0.6652 | 0.7061 | 186 | 0.3651 | 0.3931 | 222 | 0.4844 | 0.4020 | 0.0824 | 0.0000 |
| sync geo all KD fix | 184 | 183 | 0.6965 | 0.7292 | 138 | 0.3898 | 0.4051 | 138 | 0.8317 | 0.4228 | 0.0854 | 0.3235 |
| sync geo all KD warm10 fix | 198 | 197 | 0.6784 | 0.7176 | 154 | 0.3725 | 0.4031 | 171 | 0.8252 | 0.4096 | 0.0845 | 0.3311 |
| sync geo logit-only KD | 125 | 124 | 0.6471 | 0.6797 | 102 | 0.3588 | 0.3821 | 102 | 0.3923 | 0.0000 | 0.0000 | 0.3923 |

Note: rows from active runs are partial snapshots and should be refreshed after the corresponding screen exits.
References: our RGB best=0.6919; CMDistill Table I=0.740; CMDistill Table III no-KD=0.702.
