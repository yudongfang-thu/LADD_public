"""Shared augmentation policy for all OBB training lines.

All experiments use ``paired_v8_transforms`` (see ``paired_augment.py``) for train
augmentation: synchronized geometry when a teacher image is present, and the same
stack without a teacher for single-modality baselines.

Copy-paste and CutMix are disabled everywhere because the paired pipeline does not
implement them for teacher/student pairs.
"""


def apply_unified_paired_aug_policy(args) -> None:
    """Set args fields so training matches the paired augmentation pipeline."""
    args.copy_paste = 0.0
    args.cutmix = 0.0
