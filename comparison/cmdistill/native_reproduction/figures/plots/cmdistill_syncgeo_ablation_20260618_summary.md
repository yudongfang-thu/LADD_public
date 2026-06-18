| run | rows | last epoch | latest mAP50 | best mAP50 | best epoch | latest mAP50-95 | best mAP50-95 | best mAP50-95 epoch | latest KD | latest feat | latest rel | latest out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sync geo no KD | 300 | 299 | 0.6410 | 0.6853 | 173 | 0.3588 | 0.3871 | 222 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sync geo feat+rel KD | 300 | 299 | 0.6652 | 0.7061 | 186 | 0.3651 | 0.3931 | 222 | 0.4844 | 0.4020 | 0.0824 | 0.0000 |
| sync geo all KD fix | 114 | 113 | 0.6304 | 0.6894 | 92 | 0.3243 | 0.3806 | 92 | 0.9678 | 0.4993 | 0.0934 | 0.3751 |

Note: rows from active runs are partial snapshots and should be refreshed after the corresponding screen exits.
References: our RGB best=0.6919; CMDistill Table I=0.740; CMDistill Table III no-KD=0.702.
