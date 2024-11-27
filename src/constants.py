from typing import Final

from pathlib import Path

# Directories
DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
DEMO_IMAGES_DIR: Final[Path] = DATA_DIR / "images"

# Model params
MODEL_NAME: Final[str] = 'google/vit-base-patch16-224'
TARGET_IMAGE_SIZE: Final[int] = int(MODEL_NAME.split('-')[-1][-3:])
DEFAULT_PATCH_SIZE: Final[int] = int(MODEL_NAME.split('-')[-2][-2:])
