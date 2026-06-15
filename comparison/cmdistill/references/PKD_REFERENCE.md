# PKD Reference For PCC Implementation Details

PKD is not used to define CMDistill. It is only an auxiliary implementation
reference for converting Pearson-correlation feature distillation into tensor
operations when CMDistill does not specify code-level normalization dimensions.

Reference implementation:

- OpenMMLab MMRazor `PKDLoss`
- URL: `https://github.com/open-mmlab/mmrazor/blob/main/mmrazor/models/losses/pkd_loss.py`

Relevant detail:

- Normalize a feature map by channel, flattening `[N, H, W]` for each channel.
- Compute `MSE(normalized_student, normalized_teacher) / 2`.

This is consistent with the CMDistill paper's PCCFD description: normalize
feature maps to remove cross-modal magnitude effects and optimize a
Pearson-correlation feature imitation objective.
