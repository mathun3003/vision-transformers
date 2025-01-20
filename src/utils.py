import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from PIL.Image import Image
from transformers import (
    PreTrainedModel,
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
)

from src.constants import MODEL_NAME, PATCH_SIZE, TARGET_IMAGE_SIZE


@st.cache_data
def load_labels() -> dict:
    """
    load ImageNet labels
    :return: dictionary of numeric labels to categories
    """
    return torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.meta["categories"]


@st.cache_resource
def load_vit_model(
    model_name: str = MODEL_NAME, image_size: int = TARGET_IMAGE_SIZE
) -> tuple[PreTrainedModel, ViTImageProcessor]:
    """
    Loads the ViT model from HuggingFace
    :param image_size:
    :param model_name: Model name to load
    :return: HF ViT model and ViT feature extractor
    """
    # define model config
    config = ViTConfig.from_pretrained(model_name)
    config.patch_size = PATCH_SIZE
    config.output_attentions = True
    config.output_hidden_states = True
    config.interpolate_pos_encoding = False
    config.return_dict = True
    config.image_size = image_size
    # load feature extractor
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    # load model
    model = ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
    return model, feature_extractor


def img_to_patches(im: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Convert an image to patches.
    :param im: Tensor representing the image of shape [B, C, H, W]
    :param patch_size: Number of pixels per dimension of the patches
    :return: Image patches as tensor with shape [B, C, num_patches_y, num_patches_x, patch_size, patch_size]
    """
    _, _, H, W = im.shape
    # Ensure the dimensions are divisible by patch_size
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch_size"
    patches = im.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Rearrange dimensions to [B, C, num_patches_y, num_patches_x, patch_size, patch_size]
    return patches


def normalize_to_range(data: torch.Tensor, new_min: int = -1, new_max: int = 1) -> torch.Tensor:
    """
    Normalizes all tensor values to a given range.
    :param data: Input tensor of shape [B, C, H, W]
    :param new_min: Lower bound for normalization.
    :param new_max: Upper bound for normalization
    :return: Normalized tensor of shape [B, C, H, W]
    """
    data_min = data.min()
    data_max = data.max()
    normalized = (data - data_min) / (data_max - data_min) * (new_max - new_min) + new_min
    return normalized


def visualize_attention_maps(
    attentions: torch.Tensor, layer_idx: int, head_idx: int, img: Image, **kwargs
) -> plt.Figure:
    """
    Visualizing Attention Weight Matrix and (upsampled) Attention Map side by side
    :param attentions: Tensor containing the attention weihts for all layers
    :param layer_idx: selected layer index for visualization
    :param head_idx: selected head index for visualization
    :param img: input image
    :param kwargs: additional keyword arguments
    :return: figure
    """
    # select attention matrix for given layer and head
    attention_matrix = attentions[layer_idx].squeeze()[head_idx].numpy()

    fig, ax = plt.subplots(1, 3)

    # Plot the attention weight matrix
    ax[0].imshow(attention_matrix, **kwargs, cmap='jet')
    ax[0].set_title("Attention Weight Matrix", fontsize=8)

    # Generate tick positions for "CLS" and spaced-out image patches
    num_tokens = attention_matrix.shape[0]
    ticks_to_show = [0] + list(range(10, num_tokens, 20))  # CLS token and spaced-out patches
    tick_labels = ["CLS"] + [str(i) for i in ticks_to_show[1:]]

    # Configure ticks
    ax[0].set_xticks(ticks_to_show)
    ax[0].set_xticklabels(tick_labels, rotation=-30, ha="right", rotation_mode="anchor", fontsize=5)
    ax[0].set_yticks(ticks_to_show)
    ax[0].set_yticklabels(tick_labels, fontsize=5)

    # Let the horizontal axes labeling appear on top
    ax[0].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Turn on the grid
    ax[0].grid(which="minor", color="w", linestyle='--', linewidth=0.1)
    ax[0].set_xticks(np.arange(attention_matrix.shape[1] + 1) - 0.5, minor=True)
    ax[0].set_yticks(np.arange(attention_matrix.shape[0] + 1) - 0.5, minor=True)
    ax[0].tick_params(which="minor", bottom=False, left=False)

    # visualize raw attention map
    selected_attn_map = attention_matrix[0, 1:].reshape(14, 14)
    ax[1].matshow(selected_attn_map, cmap='jet')
    ax[1].set_title("Attention Map of\nCLS-to-token Sequence", fontsize=8)
    ax[1].axis('off')

    # visualize upscaled attention map as overlay image
    attn_map_scaled = (
        F.interpolate(
            torch.tensor(selected_attn_map).view(1, 1, 14, 14), size=img.size, mode="bilinear", align_corners=False
        )
        .squeeze()
        .numpy()
    )
    ax[2].imshow(img)
    ax[2].imshow(attn_map_scaled, cmap='jet', alpha=0.5)
    ax[2].set_title("Upsampled Attention Map of\nCLS-to-token-Sequence", fontsize=8)
    ax[2].axis('off')

    return fig


def compute_scaled_qkv(
    image_size: tuple[int, int],
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    channel: int,
    scaling_factor: int,
) -> dict[str, torch.Tensor]:
    """
    Computes scaled visualizations for query, key, value, and self-attention for a given channel.

    Args:
        image_size (tuple): The size of the input image (height, width).
        queries (torch.Tensor): The query tensor with shape [1, num_tokens, num_channels].
        keys (torch.Tensor): The key tensor with shape [1, num_tokens, num_channels].
        values (torch.Tensor): The value tensor with shape [1, num_tokens, num_channels].
        channel (int): The specific channel to compute visualizations for.
        scaling_factor (int): Scaling factor for self-attention computation.

    Returns:
        dict: A dictionary containing the scaled visualizations for query, key, value, and self-attention.
    """
    tensors = {"query": queries, "key": keys, "value": values}
    visualizations = {}

    for tensor_name, tensor in tensors.items():
        tensor_data = tensor.squeeze(0)[1:, channel].view(1, 1, 14, 14)  # Skip CLS token
        tensor_resized = (
            F.interpolate(tensor_data, size=image_size, mode="bilinear", align_corners=False)
            .squeeze(0)
            .detach()
            .cpu()
            .view(*image_size, 1)
            .numpy()
        )
        visualizations[tensor_name] = normalize_to_range(tensor_resized)

    # Compute self-attention
    sa = (
        (
            torch.softmax(
                (queries.squeeze(0)[1:, channel].unsqueeze(1) @ keys.squeeze(0)[1:, channel].unsqueeze(1).T)
                / scaling_factor,
                dim=0,
            )
            @ values.squeeze(0)[1:, channel].unsqueeze(1)
        )
        .squeeze(1)
        .view(1, 1, 14, 14)
    )
    sa_resized = F.interpolate(sa, size=image_size, mode="bilinear", align_corners=False)
    visualizations["self_attention"] = normalize_to_range(
        sa_resized.squeeze(0).detach().cpu().view(*image_size, 1).numpy()
    )

    return visualizations


def visualize_qkv(
    image: Image,
    visualizations: dict,
    selected_channels: list,
    layer_index: int,
) -> plt.Figure:
    """
    Visualizes the computed query, key, value, and self-attention visualizations.

    Args:
        image (Image): The input image to overlay visualizations on.
        visualizations (dict): A dictionary of computed visualizations for each channel.
        selected_channels (list): List of channels to visualize.
        layer_index (int): Index of the layer being visualized.

    Returns:
        plt.Figure: A matplotlib figure containing the visualizations.
    """
    # Set up the plot grid
    fig, axes = plt.subplots(nrows=len(selected_channels), ncols=5, figsize=(25, 15))
    colormap = "jet"

    # Set the overall figure title
    fig.suptitle(
        f"Query, Key, and Value Visualization for CLS token of Layer {layer_index} (single head)", fontsize=20, y=0.95
    )

    # Set column titles
    column_titles = ["Query", "Key", "Value", "Self-Attention (normalized)", "Original Image"]
    for col, title in enumerate(column_titles):
        fig.text(
            x=0.18 + col * 0.14,
            y=0.9,
            s=title,
            ha="center",
            fontsize=16,
        )

    for idx, channel in enumerate(selected_channels):
        # Add row title with the channel number
        axes[idx, 0].text(-50, image.size[1] // 2, f"Channel {channel}", rotation=90, va="center", fontsize=14)

        # Create visualizations for each type
        for col, (_, data) in enumerate(list(visualizations[channel].items()) + [("original", None)]):
            axes[idx, col].imshow(image)
            if data is not None:
                axes[idx, col].imshow(data, cmap=colormap, alpha=0.5, vmin=-1, vmax=1)
            axes[idx, col].axis("off")

    # Add a colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes((0.82, 0.15, 0.01, 0.7))
    fig.colorbar(axes[0, 0].images[-1], cax=cbar_ax, orientation="vertical", label="Normalized Values in range [-1, 1]")

    return fig


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
        attn_fused = 0.5 * attn_fused + 0.5 * torch.eye(num_tokens, device=attn_fused.device).unsqueeze(0)

        # Normalize attention matrix
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

        # Update rollout
        rollout = torch.matmul(attn_fused, rollout)

        # Extract CLS-to-input-token attention (exclude CLS token itself)
        cls_to_input = rollout[:, 0, 1:].detach()
        cls_rollouts.append(cls_to_input)

    return cls_rollouts


def visualize_attention_rollout(
    image: Image, attention_rollouts: list[torch.Tensor], patch_size: int, number_of_layers: int
):
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
        attn_map_resized = (
            F.interpolate(cls_attn_map_view, size=image.size, mode="bilinear", align_corners=False)
            .squeeze()
            .detach()
            .cpu()
            .view(*image.size, 1)
            .numpy()
        )

        # Overlay attention map on the axis
        ax.imshow(img_array)
        ax.imshow(attn_map_resized, cmap='jet', alpha=0.5)
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.axis("off")
    plt.tight_layout()
    return fig
