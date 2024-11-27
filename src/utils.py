import numpy as np
import numpy.typing as npt
import streamlit as st
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from transformers import ViTImageProcessor, ViTConfig, ViTForImageClassification, PreTrainedModel

from src.constants import MODEL_NAME, DEFAULT_PATCH_SIZE, TARGET_IMAGE_SIZE


@st.cache_data
def load_labels() -> dict:
    """
    load ImageNet labels
    :return: dictionary of numeric labels to categories
    """
    return torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.meta["categories"]


@st.cache_resource
def load_hf_model(model_name: str = MODEL_NAME) -> tuple[PreTrainedModel, ViTImageProcessor]:
    """
    Loads the ViTImageProcessor model and returns it.
    :param model_name:
    :return:
    """
    config = ViTConfig.from_pretrained(model_name)
    config.patch_size = DEFAULT_PATCH_SIZE
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    config.image_size = TARGET_IMAGE_SIZE
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


def generate_attention_maps(attentions: list, method: str = 'mean') -> list[torch.Tensor]:
    """
    Generates attention maps per layer from ViT for the given method.
    :param attentions: List of attentions from each layer of shape: (batch_size, num_heads, sequence_length, sequence_length)
    :param method: Either mean, max, or min.
    :return: Averaged/Min/Max Attention maps
    """

    """
    NOTE:
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    # Select the attention for the [CLS] token
    cls_attentions = [att[:, :, 0, 1:] for att in attentions]  # Exclude [CLS]-[CLS] attention
    if method == 'mean':
        # Average over all heads
        cls_attentions_avg = [att.mean(dim=1) for att in cls_attentions]
        return cls_attentions_avg
    elif method == 'max':
        # take the max of all heads
        cls_attentions_max = [att.max(dim=1).values for att in cls_attentions]
        return cls_attentions_max
    elif method == 'min':
        cls_attentions_min = [att.min(dim=1).values for att in cls_attentions]
        return cls_attentions_min
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
        attention = attention[0].reshape(grid_size, grid_size).numpy()
        # TODO: check value range after normalization (is it in [0,1]?)
        attention = attention / attention.max()  # Normalize for better visualization

        # Resize attention to image size
        attention_resized = np.array(Image.fromarray(attention).resize(image.size, Image.BILINEAR))

        # Overlay attention on image
        ax.imshow(image_np)
        ax.imshow(attention_resized, cmap="jet", alpha=0.5)
        ax.axis("off")
        ax.set_title(f"Layer {idx + 1}")

    plt.tight_layout()
    return fig

