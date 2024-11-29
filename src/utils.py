from typing import Callable

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from timm import create_model
from timm.models.vision_transformer import Attention, VisionTransformer

from src.constants import MODEL_NAME, DEFAULT_PATCH_SIZE, TARGET_IMAGE_SIZE


def forward_wrapper(attn_obj: Attention) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Decorator function to catch attentions during model inference.
    :param attn_obj: Attention object.
    :return: Overwritten forward function
    """
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Redefine forward function
        :param x: input tensor
        :return: output tensor
        """
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 1:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return forward


@st.cache_data
def load_labels() -> dict:
    """
    load ImageNet labels
    :return: dictionary of numeric labels to categories
    """
    return torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.meta["categories"]


@st.cache_resource
def load_vit_model(model_name: str = MODEL_NAME, patch_size: int = DEFAULT_PATCH_SIZE) -> VisionTransformer:
    """
    Loads the ViT model using timm library
    :param model_name: Model name to load
    :param patch_size: Patch size for ViT
    :return: timm model
    """
    model = create_model(model_name, patch_size=patch_size, pretrained=True)
    for block in model.blocks:
        # overwrite forward function for every attention layer
        block.attn.forward = forward_wrapper(block.attn)
    return model


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


def generate_attention_maps(model: VisionTransformer, method: str = 'max') -> list[torch.Tensor]:
    """
    Generates attention maps per layer from ViT for the given method.
    :param model: Vanilla Vision Transformer model.
    :param method: Either mean, max, or min.
    :return: Averaged/Min/Max Attention maps
    """

    """
    NOTE:
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    # Select the attention for the [CLS] token
    attentions = [block.attn.attn_map for block in model.blocks]
    if method == 'mean':
        # Average over all heads
        return [attention_map.mean(dim=1).squeeze(0).detach() for attention_map in attentions]
    elif method == 'max':
        # take the max of all heads
        return [attention_map.max(dim=1).values.squeeze(0).detach() for attention_map in attentions]
    elif method == 'min':
        return [attention_map.min(dim=1).values.squeeze(0).detach() for attention_map in attentions]
    else:
        raise ValueError(f"Method {method} not supported")


# Visualize attention maps
def visualize_attention_maps(image: Image, attention_maps: list[torch.Tensor]) -> plt.Figure:
    """
    Visualizes attention maps of different ViT layers.
    :param image: Input image
    :param attention_maps: list of attention maps per layer.
    :return: Figure
    """
    image_np = np.array(image)

    # Set the desired number of rows and columns
    fig, axes = plt.subplots(4, 3, figsize=(15, 15))

    # Flatten the axes array to iterate over
    axes = axes.flatten()

    for idx, attention in enumerate(attention_maps):
        ax = axes[idx]

        # Reshape attention to image grid size (e.g., 14x14 for ViT Base)
        grid_size = int(attention.shape[-1] ** 0.5)
        # attention = attention[0].reshape(grid_size, grid_size).numpy()
        # # TODO: check value range after normalization (is it in [0,1]?)
        # attention = attention / attention.max()  # Normalize for better visualization
        #
        # # Resize attention to image size
        # attention_resized = np.array(Image.fromarray(attention).resize(image.size, Image.BILINEAR))

        # Overlay attention on image
        ax.imshow(image_np)
        ax.imshow(attention, cmap="jet", alpha=0.5)
        ax.axis("off")
        ax.set_title(f"Layer {idx + 1}")

    plt.tight_layout()
    return fig


def compute_attention_rollout(attn_maps: list[torch.Tensor], grid_size: int = DEFAULT_PATCH_SIZE, image_size: tuple[int, int] = TARGET_IMAGE_SIZE) -> list[torch.Tensor]:
    """
    Computes the attention rollout over all layers for the CLS token.
    :param attn_maps: Raw attention maps per layer.
    :param grid_size: Grid size for upsampling.
    :param image_size: Image size for upsampling.
    :return: Attention rollout per layer for CLS token.
    """
    attn_rollout: list[torch.Tensor] = []
    I = torch.eye(attn_maps[0].shape[-1])  # Identity matrix
    prod = I
    for i, attn_map in enumerate(attn_maps):
        prod = prod @ (attn_map + I)  # Product of attention maps with identity matrix
        prod = prod / prod.sum(dim=-1, keepdim=True)  # Normalize
        # Keep only for the CLS token
        attn_rollout.append(prod)
    return attn_rollout


def get_attention_rollout_per_layer(model, input_tensor):
    """
    Compute attention rollout for a Vision Transformer model layer by layer.

    Args:
        model: The Vision Transformer model (from timm).
        input_tensor: Input tensor of shape (B, C, H, W).

    Returns:
        List of cumulative attention maps for each layer.
    """
    # Forward hook to capture attention maps
    attention_maps = []

    def hook(module, input, output):
        avg_attention = output.mean(dim=1)  # Average attention across heads
        attention_maps.append(avg_attention)

    # Register hooks on all attention layers
    hooks = []
    for block in model.blocks:
        hooks.append(block.attn.attn_drop.register_forward_hook(hook))

    # Forward pass
    _ = model(input_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Process attention maps
    rollout = torch.eye(attention_maps[0].size(-1)).to(input_tensor.device)  # Identity matrix
    rollout_per_layer = []
    for attention in attention_maps:
        attention += torch.eye(attention.size(-1)).to(input_tensor.device)  # Add residual connection
        attention /= attention.sum(dim=-1, keepdim=True)  # Normalize
        rollout = torch.matmul(attention, rollout)  # Update cumulative attention
        rollout_per_layer.append(rollout[:, 0, 1:])  # Extract [CLS]-to-patches attention

    return rollout_per_layer


def visualize_attention_rollout(image, attention_rollouts, patch_size):
    """
    Visualize attention rollout per layer as overlays on the original image.

    Args:
        image: The original image (PIL Image).
        attention_rollouts: List of cumulative attention maps for each layer.
        patch_size: The size of each patch in the Vision Transformer.
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    fig, axes = plt.subplots(4, 3, figsize=(15, 15))
    axes = axes.flatten()  # Flatten for easy iteration

    # Resize attention maps to image size for overlay
    for layer_idx, (ax, attn_map) in enumerate(zip(axes, attention_rollouts)):
        # select CLS attention maps
        cls_attn_map = attn_map[0, 1:]
        # reshape CLS attention map
        cls_attn_map_view = cls_attn_map.view(1, 1, int(img_h / patch_size), int(img_w / patch_size))
        # upsample CLS attention map to image size
        attn_map_resized = F.interpolate(
            cls_attn_map_view, size=image.size, mode="bilinear", align_corners=False
        ).squeeze().detach().cpu().view(*image.size, 1).numpy()

        # Overlay attention map on the axis
        ax.imshow(img_array)
        ax.imshow(attn_map_resized, cmap='jet', alpha=0.5)
        ax.set_title(f"Attention Rollout - Layer {layer_idx + 1}")
        ax.axis("off")
    plt.tight_layout()
    return fig
