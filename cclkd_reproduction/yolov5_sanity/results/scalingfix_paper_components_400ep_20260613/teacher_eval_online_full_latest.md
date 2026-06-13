# Online Teacher Evaluation Snapshot

Snapshot time: `2026-06-14 02:33:14 +0800`.

Evaluated `paper_full` checkpoint `last.pt` teacher branch at checkpoint epoch `214` on RGB validation data.

| model/branch | epoch | modality | AP50 | AP |
| --- | ---: | --- | ---: | ---: |
| Full run online teacher | 214 | RGB | 0.80073 | 0.42392 |
| Full run student | 214 | SAR | 0.61927 | 0.34123 |
| Det-only student | 214 | SAR | 0.60733 | 0.33427 |
| Independent RGB YOLOv5x baseline | 399 | RGB | 0.86506 | 0.52414 |

Teacher-student AP gap: `0.08269`. Full student gain over det-only at the same epoch: `0.00696`. Approximate recovered teacher-det gap: `0.07763`.

Interpretation: the online teacher is substantially stronger than the SAR student, but lower than the independent RGB YOLOv5x 400epoch upper baseline. The current KD path transfers only a small fraction of the available teacher-det gap.
