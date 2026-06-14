# YOLOv5x CCLKD Component Decomposition Snapshot

Generated from compact local archive. Exact same-epoch det-only deltas are used when available.

## Latest Runs

| run | epoch | AP | delta AP vs det | KD/det | LLD | FLD | RLD | CCL | ATKD share | CCL share | COP+ | T mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Det-only | 399 | 0.439880 |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |  |  |
| ATKD-only | 287 | 0.389940 | 0.013140 | 0.057651 | 0.008879 | 0.033811 | 0.050182 | 0.000000 | 1.000000 | 0.000000 | 0.988012 | 2.830132 |
| CCL-only | 399 | 0.443480 | 0.003600 | 0.556275 | 0.000000 | 0.000000 | 0.000000 | 0.693987 | 0.000000 | 1.000000 | 0.993671 | 2.784254 |
| Full CCLKD | 392 | 0.444830 | 0.007950 | 0.613547 | 0.006232 | 0.034696 | 0.033274 | 0.693717 | 0.096627 | 0.903373 | 0.992846 | 2.788691 |

## Exact Milestones

| epoch | det AP | ATKD AP / delta | CCL AP / delta | Full AP / delta | Full-ATKD | Full-CCL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.325200 | 0.333830 / 0.008630 | 0.329360 / 0.004160 | 0.332550 / 0.007350 | -0.001280 | 0.003190 |
| 250 | 0.356250 | 0.365220 / 0.008970 | 0.361160 / 0.004910 | 0.363600 / 0.007350 | -0.001620 | 0.002440 |
| 300 | 0.384160 | pending / pending | 0.392270 / 0.008110 | 0.393400 / 0.009240 | pending | 0.001130 |
| 350 | 0.414710 | pending / pending | 0.422230 / 0.007520 | 0.424150 / 0.009440 | pending | 0.001920 |
| 399 | 0.439880 | pending / pending | 0.443480 / 0.003600 | pending / pending | pending | pending |

## Immediate Diagnostic Reading

- ATKD-only currently has weak but consistent positive AP gain. Its weighted KD/det ratio is low, so ATKD may be under-coupled to detector optimization.
- CCL-only reaches 399 epochs but has very small final AP gain while weighted KD/det is high, so CCL is a low-efficiency signal in the current implementation.
- Full CCLKD is positive vs det-only but is not consistently better than ATKD-only at common 200/250 epochs; this supports a weak/negative CCL synergy diagnosis until later milestones prove otherwise.
- These are descriptive diagnostics only. Mechanism-level proof still requires offline gradient/cosine and CCL pos-vs-neg similarity probes.
