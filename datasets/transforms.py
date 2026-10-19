"""
Data augmentation transforms.

Key constraint from EGMA paper:
  SPATIAL augmentations (flip, rotation, crop) must NOT be applied when gaze
  heatmaps are in use, because they destroy the gaze-patch correspondence.
  Only photometric (contrast/brightness) augmentations are safe.
"""

from torchvision import transforms


def get_jepa_transform(img_size=224, is_train=True):
    """
    Standard I-JEPA image transform.
    For non-gaze training (CheXpert Stage 1), spatial augmentations are OK.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


def get_gaze_safe_transform(img_size=224, is_train=True,
                             img_mean=0.5862, img_std=0.2795):
    """
    Photometric-only transform — safe when gaze heatmaps are present.
    No spatial ops that would break gaze-patch correspondence.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[img_mean], std=[img_std]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[img_mean], std=[img_std]),
        ])


def get_heatmap_transform(grid_size=14):
    """
    Resize gaze heatmap (any input size) to the ViT patch grid (grid_size x grid_size).
    Returns a float tensor in [0, 1].
    """
    return transforms.Compose([
        transforms.Resize((grid_size, grid_size)),
        transforms.ToTensor(),   # (1, G, G), values in [0, 1] if input was uint8 PIL
    ])


def get_chexpert_eval_transform(img_size=224, img_mean=0.5862, img_std=0.2795):
    """No augmentation for evaluation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[img_mean], std=[img_std]),
    ])
