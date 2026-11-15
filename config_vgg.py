# config_vgg.py
import os

VOC_ROOT   = "data/dataset"
VOC_CLASSES = ('background', 'license_plate', 'car')

NUM_CLASSES = len(VOC_CLASSES)   # 3

# Обучение
BATCH_SIZE   = 8
NUM_WORKERS  = 4
LR           = 1e-3
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS       = 30
LR_STEPS     = [15, 25]
LR_GAMMA     = 0.1

# Loss
OVERLAP_THRESHOLD = 0.5
NEG_POS_RATIO     = 3


# Сохранение
RUN_DIR = os.path.join("runs", "vgg_ssd_640x360")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "samples"), exist_ok=True)


FEATURE_MAPS = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
MIN_SIZES  = [30, 60, 111, 162, 213, 264]
MAX_SIZES  = [60, 111, 162, 213, 264, 315]
ASPECT_RATIOS =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
VARIANCE   = [0.1, 0.2]
CLIP       = True

PRIOR_CFG = {
    "feature_maps": FEATURE_MAPS,
    "min_sizes":    MIN_SIZES,
    "max_sizes":    MAX_SIZES,
    "aspect_ratios":ASPECT_RATIOS,
    "variance":     VARIANCE,
    "clip":         CLIP,
}