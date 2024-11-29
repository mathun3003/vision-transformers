from typing import Final

from pathlib import Path

# Directories
DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
DEMO_IMAGES_DIR: Final[Path] = DATA_DIR / "images"

# Model params
MODEL_NAME: Final[str] = 'vit_large_patch16_224'  # 'google/vit-base-patch16-224'
TARGET_IMAGE_SIZE: Final[int] = int(MODEL_NAME.split('_')[-1][-3:])
DEFAULT_PATCH_SIZE: Final[int] = int(MODEL_NAME.split('_')[-2][-2:])
