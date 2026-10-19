"""
Medical vocabulary for logit-lens style semantic projection.

Each term is a "concept" we project patches onto. The list below covers
CheXpert's 14 official classes plus auxiliary anatomical/finding terms
that give richer language coverage of chest X-rays.

Order matters — used as the fixed index for word distributions.
"""

CHEXPERT_14 = [
    "no finding",
    "enlarged cardiomediastinum",
    "cardiomegaly",
    "lung opacity",
    "lung lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural effusion",
    "pleural other",
    "fracture",
    "support devices",
]

EXTRA_CONCEPTS = [
    "normal lung",
    "interstitial pattern",
    "infiltrate",
    "nodule",
    "mass",
    "calcification",
    "hyperinflation",
    "endotracheal tube",
    "pacemaker wire",
    "rib",
    "spine",
    "diaphragm",
    "heart",
    "vessel",
    "hilum",
    "trachea",
]

MEDICAL_VOCAB = CHEXPERT_14 + EXTRA_CONCEPTS  # 30 terms total

# CheXpert label indices in the vocab (first 14)
CHEXPERT_LABEL_IDX = list(range(len(CHEXPERT_14)))


def vocab_size() -> int:
    return len(MEDICAL_VOCAB)


def vocab_to_prompts() -> list[str]:
    """Wrap each concept in a CXR-style prompt to improve text encoder quality."""
    return [f"a chest x-ray showing {c}" for c in MEDICAL_VOCAB]
