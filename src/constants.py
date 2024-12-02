from typing import Final

from pathlib import Path

from yarl import URL

# Directories
DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
DEMO_IMAGES_DIR: Final[Path] = DATA_DIR / "images"

# Model params
MODEL_NAME: Final[str] = 'google/vit-base-patch16-224'
TARGET_IMAGE_SIZE: Final[int] = int(MODEL_NAME.split('-')[-1][-3:])
PATCH_SIZE: Final[int] = int(MODEL_NAME.split('-')[-2][-2:])

# Image URLs
ATTN_BLOCK_IMG_URL: Final[URL] = URL(
    'https://theaisummer.com/static/aa65d942973255da238052d8cdfa4fcd/7d4ec/the-transformer-block-vit.png'
)
VIT_GIF_URL: Final[URL] = URL(
    'https://research.google/blog/transformers-for-image-recognition-at-scale/'
)
