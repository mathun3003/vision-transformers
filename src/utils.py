import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from transformers import ViTConfig, ViTImageProcessor, ViTForImageClassification, PreTrainedModel

from src.constants import MODEL_NAME, DEFAULT_PATCH_SIZE, TARGET_IMAGE_SIZE


@st.cache_data
def load_labels() -> dict:
    """
    load ImageNet labels
    :return: dictionary of numeric labels to categories
    """
    return torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.meta["categories"]


@st.cache_resource
def load_vit_model(
        model_name: str = MODEL_NAME,
        patch_size: int = DEFAULT_PATCH_SIZE,
        image_size: int = TARGET_IMAGE_SIZE
) -> tuple[PreTrainedModel, ViTImageProcessor]:
    """
    Loads the ViT model from HuggingFace
    :param image_size:
    :param model_name: Model name to load
    :param patch_size: Patch size for ViT
    :return: HF ViT model and ViT feature extractor
    """
    # define model config
    config = ViTConfig.from_pretrained(model_name)
    config.patch_size = patch_size
    config.output_attentions = True,
    config.output_hidden_states = True,
    config.interpolate_pos_encoding = False,
    config.return_dict = True
    config.image_size = image_size
    # load feature extractor
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    # load model
    model = ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
    return model, feature_extractor


def img_to_patches(_im: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Convert an image to patches.
    :param _im: Tensor representing the image of shape [B, C, H, W]
    :param patch_size: Number of pixels per dimension of the patches
    :return: Image patches as tensor with shape [B, C, num_patches_y, num_patches_x, patch_size, patch_size]
    """
    B, C, H, W = _im.shape
    # Ensure the dimensions are divisible by patch_size
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch_size"
    patches = _im.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Rearrange dimensions to [B, C, num_patches_y, num_patches_x, patch_size, patch_size]
    return patches


def compute_attention_rollout(attn_maps: list[torch.Tensor], fusion_method: str = 'mean') -> list[torch.Tensor]:
    """
    Computes attention rollout for vision transformers, focusing on the CLS-to-input-token matrix.

    :param attn_maps: List of attention matrices (B, H, N, N), where B = batch size, H = number of attention heads,
    N = number of tokens (including CLS).
    :param fusion_method: Fusion method across heads in multi-head attention block.

    :returns: list of torch.Tensor: Rollout matrices for each layer, focusing on CLS-to-input-token. Each matrix has
    shape (B, N-1), excluding the CLS token itself.
    """
    batch_size: int = attn_maps[0].size(0)
    # Total tokens including CLS
    num_tokens: int = attn_maps[0].size(-1)
    cls_rollouts: list[torch.Tensor] = []

    # Initialize rollout with an identity matrix for the CLS token
    rollout: torch.Tensor = torch.eye(num_tokens, device=attn_maps[0].device).unsqueeze(0).repeat(batch_size, 1, 1)

    for attn_map in attn_maps:
        # Average attention heads
        if fusion_method == 'mean':
            attn_fused = attn_map.mean(dim=1)
        elif fusion_method == 'max':
            attn_fused = attn_map.max(dim=1).values
        elif fusion_method == 'min':
            attn_fused = attn_map.min(dim=1).values
        else:
            raise ValueError(f"Fusion method {fusion_method} not supported")
        # Add skip connection (identity matrix)
        attn_fused = attn_fused + torch.eye(num_tokens, device=attn_fused.device).unsqueeze(0)

        # Normalize attention matrix
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

        # Update rollout
        rollout = torch.matmul(attn_fused, rollout)

        # Extract CLS-to-input-token attention (exclude CLS token itself)
        cls_to_input = rollout[:, 0, 1:].detach()
        cls_rollouts.append(cls_to_input)

    return cls_rollouts


def visualize_attention_rollout(image: Image, attention_rollouts: list[torch.Tensor], patch_size: int,
                                number_of_layers: int):
    """
    Visualize attention rollout per layer as overlays on the original image.

    :param image: The original image (PIL Image).
    :param attention_rollouts: List of cumulative attention maps for each layer.
    :param patch_size: The size of each patch in the Vision Transformer.
    :param number_of_layers: Number of layers in the Vision Transformer.
    :return: figure
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    n_cols: int = 3
    fig, axes = plt.subplots(number_of_layers // n_cols, 3, figsize=(15, 15))
    axes = axes.flatten()

    # Resize attention maps to image size for overlay
    for layer_idx, (ax, cls_attn_map) in enumerate(zip(axes, attention_rollouts)):
        # reshape CLS attention map
        cls_attn_map_view = cls_attn_map.view(1, 1, int(img_h / patch_size), int(img_w / patch_size))
        # upsample CLS attention map to image size
        attn_map_resized = F.interpolate(
            cls_attn_map_view, size=image.size, mode="bilinear", align_corners=False
        ).squeeze().detach().cpu().view(*image.size, 1).numpy()

        # Overlay attention map on the axis
        ax.imshow(img_array)
        ax.imshow(attn_map_resized, cmap='jet', alpha=0.5)
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.axis("off")
    plt.tight_layout()
    return fig
