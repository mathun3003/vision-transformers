import streamlit as st
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn

from src.constants import (
    ATTN_BLOCK_IMG_URL,
    DEMO_IMAGES_DIR,
    PATCH_SIZE,
    TARGET_IMAGE_SIZE,
    VIT_GIF_URL,
)
from src.utils import (
    compute_attention_rollout,
    compute_scaled_qkv,
    img_to_patches,
    load_labels,
    load_vit_model,
    visualize_attention_rollout,
    visualize_qkv, visualize_attention_maps,
)

# load ImageNet labels
labels = load_labels()

st.title("🤖 Vision Transformers - Explained")

st.text("Enjoy this short demonstration of Vision Transformers!")

st.subheader("From Image to Classification")

# display ViT GIF
st.video(
    './data/vit-gif.mp4',
    loop=True,
    autoplay=True,
    muted=True,
)
st.markdown(f'<a href="{str(VIT_GIF_URL)}" style="color: lightgrey;">GIF Source</a>', unsafe_allow_html=True)

# display pages
st.divider()
st.subheader("🖼️ Visualize Patches")

st.text(
    """
    Select an object to visualize how Vision Transformers patchify an image.
"""
)

options_to_img = {
    "Catamaran": DEMO_IMAGES_DIR / 'catamaran.jpeg',
    "Airplane": DEMO_IMAGES_DIR / 'airplane.jpg',
    "Steam Locomotive": DEMO_IMAGES_DIR / 'steam_locomotive.jpeg',
}
# display selection options
selection = st.pills("Objects", options_to_img.keys(), selection_mode="single")
# wait for an image selection
if not selection:
    st.stop()

img_path = options_to_img[selection].as_posix()
# load image from path
img_raw = Image.open(img_path).resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))

# display image dimensions
st.text("Image dimensions:")
st.latex(fr"(H,W)={img_raw.size}")

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))]
)
# apply torch transformation, remove batch dimension
img = transform(img_raw).unsqueeze(0)

st.markdown("Number of image patches $N$:")
# display the number of image patches
st.latex(
    fr"N=HW/P^2={img_raw.size[0]}\cdot{img_raw.size[1]}/{PATCH_SIZE}^2="
    fr"{int((img_raw.size[0] * img_raw.size[1]) / PATCH_SIZE ** 2)}"
)

left_col, right_col = st.columns(2)

with left_col:
    # display selected image
    st.text("Selected image")
    st.image(img_path, use_container_width=True)

with right_col:
    st.text("Image Patches")

    with st.spinner('Splitting image into patches', _cache=True):
        # split image into patches
        img_patches = img_to_patches(img, PATCH_SIZE)
        # Visualize patches for the first image in the batch
        _, _, num_patches_y, num_patches_x, _, _ = img_patches.shape
        fig, axs = plt.subplots(num_patches_y, num_patches_x, figsize=(5, 5))

        for i in range(num_patches_y):
            for j in range(num_patches_x):
                # Extract the patch (first image in the batch)
                # Convert to [H, W, C] for visualization
                patch = img_patches[0, :, i, j].permute(1, 2, 0).numpy()
                # normalize patch values to [0, 1]
                patch = (patch + 1) / 2
                axs[i, j].imshow(patch)
                axs[i, j].axis("off")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

st.divider()


# pylint: disable=unused-argument
def hook(module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
    """
    Custom PyTorch hook to catch query, key, and value tensors during inference.
    :param module: PyTorch model's module where the tensors should be caught
    :param input_tensor: Input tensor
    :param output_tensor: Output tensor (unused argument but model expects it)
    :return: None
    """
    global queries, keys, values
    queries = module.query(input_tensor[0])
    keys = module.key(input_tensor[0])
    values = module.value(input_tensor[0])


with st.spinner("Performing model prediction", _cache=True):
    model, feature_extractor = load_vit_model(image_size=img_raw.size[0])
    # register hook
    queries, keys, values = torch.Tensor, torch.Tensor, torch.Tensor
    layer_idx: int = 4
    attn_layer = model.vit.encoder.layer[layer_idx].attention.attention
    hook_handle = attn_layer.register_forward_hook(hook)

# get model outputs
inputs = feature_extractor(images=img_raw, return_tensors="pt", do_resize=True, size=TARGET_IMAGE_SIZE)
with torch.no_grad():
    outputs = model(**inputs)
    hook_handle.remove()

# get prediction
logits = outputs.logits
predicted_class = logits.argmax(dim=1).item()
predicted_label = labels[predicted_class]
prediction_proba = round(torch.max(F.softmax(logits, dim=1), dim=1).values.item(), 2)

# NOTE: Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads
attentions = outputs.attentions

num_layers = len(attentions)
_, num_heads, num_patches, sequence_length = attentions[-1].shape

st.subheader("🎛️ Model Parameters and Embedding Dimensions")
st.write(
    f"""
    **Model Prediction**\n
    Predicted label: {predicted_label.capitalize()}\n
    Prediction confidence: {prediction_proba}\n
    **Internal Model Parameters**\n
    Number of Attention Layers: {num_layers}\n
    Number of Self-attention Heads: {num_heads}\n
    Number of Tokens: {num_patches - 1} + 1 CLS token\n
    Sequence Length: {sequence_length}
"""
)

st.divider()

st.subheader("👁️‍🗨️ Attention Weight Matrix & Attention Map")

st.markdown(r"Queries, keys, and values are obtained by a linear projection of the input embeddings to a specific "
            r"attention block. The dot product $\mathbf{QK}^T \in \mathbb{R}^{(N+1) \times (N+1)}$ yields the "
            r"so-called attention score for each combination of tokens. The result is the attention weight matrix, "
            r"which contains all attention scores for head $h \in [1,...,H]$ in layer $l \in [1,...,L]$. "
            r"The first row represents the CLS-to-patch attentions. As this sequence has the same length as number of "
            r"patches, one can reshape this sequence to 2D. The resulting image is of lower resolution but can be "
            r"upsampled to match the input image size.")

st.latex(r"A^{(l)}_h = \text{softmax}(\frac{\mathbf{Q}^{(l)}_h {\mathbf{K}^{(l)}_h}^T}{\sqrt{d}})\;")

# visualize attention weight matrix and attention maps
selected_layer = st.select_slider(label=r"Encoder Layer $l$", options=range(1, 13), value=1) - 1
selected_head = st.select_slider(label=r"Selected Head $h$ in layer $l$", options=range(1, 13), value=1) - 1
with st.spinner("Visualizing Attention", _cache=True):
    attn_weight_matrix = visualize_attention_maps(attentions, layer_idx=selected_layer, head_idx=selected_head, img=img_raw)
    st.pyplot(attn_weight_matrix)

st.subheader("🔍 Queries, Keys, and Values")
st.text(
    "Since the image is split into patches, the queries, keys, and values represent the transformed and embedded "
    "input image inside the transformer. One can visualize the query, key, and value image for a given layer, "
    "head, and embedding dimension."
)
st.markdown(
    r"For each self-attention layer, the input embedding $\mathbf{z}^{(l)} \in \mathbb{R}^{(N+1) \times d}$ goes "
    r"through three linear projection layers with the weight matrices "
    r"$W_Q \in \mathbb{R}^{d \times d}, \; W_K \in \mathbb{R}^{d \times d},\; W_V \in \mathbb{R}^{d \times d}$ "
    r"to compute the queries, keys, and values. "
)
st.latex(
    r"\mathbf{Q}^{(l)}=\mathbf{z}^{(l)}W_Q,\; \mathbf{K}^{(l)}=\mathbf{z}^{(l)}W_K,\; \mathbf{V}^{(l)}=\mathbf{z}^{("
    r"l)}W_V \in \mathbb{R}^{(N+1) \times d}"
)
st.markdown(
    r"Queries and keys are used to generate the so-called attention weights in self-attention $\mathbf{QK}^T$. "
    r"Then, the self-attention is computed as follows:"
)
st.latex(
    r"\text{SA}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\text{softmax}(\frac{\mathbf{QK}^T}{\sqrt{d}})\mathbf{V} \in "
    r"\mathbb{R}^{(N+1) \times d}"
)

# visualize queries, keys, and values for channel dimensions
selected_channels = [21, 22, 26, 42]
with st.spinner("Computing Queries, Keys, and Values"):
    embedding_dim: int = queries.shape[-1]
    scaled_qkv = {
        channel: compute_scaled_qkv(
            img_raw.size, torch.tensor(queries), torch.tensor(keys), torch.tensor(values), channel, embedding_dim
        )
        for channel in selected_channels
    }
    st.pyplot(visualize_qkv(img_raw, scaled_qkv, selected_channels, layer_idx))

st.divider()

st.subheader("🔁 Attention Rollout")
st.text(
    "In order to see how the attention flows through the network, multiplying the attention weights from each "
    "attention block recursively at each layer results in the so-called attention rollout. In a Multi-head "
    "Attention Layer, the self-attentions are fused together to a single attention weight matrix using either the "
    "maximum, minimum, or mean across the heads' attentions."
)
st.markdown(
    r"For $L$ layers, the attention rollout $\tilde{A}^{(l)}$ at layer $l \in \{1,…,L\}$ is recursively defined as:"
)
st.latex(
    r"""
    \tilde{A}^{(l)}=
    \begin{cases}
    A^{(l)}\tilde{A}^{(l-1)} & \text{if } l > 1 \\
    A^{(l)} & \text{if } l=1
    \end{cases}
"""
)
st.markdown(
    r"Where $A^{(l)}$ is the raw attention of layer $l$. To incorporate the skip connections around the "
    r"Multi-head Attention Layer, $A^{(l)}$ is computed as the average of the input activations and the attention "
    r"weights $W_{attn}$:"
)
st.latex(r"A^{(l)}=0.5 W_{attn}^{(l)} + 0.5I^{(l)}")

fusion_method = st.selectbox(
    label="Fusion Method", options=["Mean", "Min", "Max"], help="Fusion method across heads in a MHA Layer"
)
left_col, right_col = st.columns(spec=[0.7, 0.3], vertical_alignment="center")
with left_col:
    with st.spinner("Computing Attention Rollout"):
        attention_rollout = compute_attention_rollout(attentions, fusion_method=fusion_method.lower())
        st.pyplot(visualize_attention_rollout(img_raw, attention_rollout, PATCH_SIZE, num_layers))
with right_col:
    st.image(str(ATTN_BLOCK_IMG_URL), caption="Encoder Layer in ViT")
    st.markdown(
        f'<a href="{str(ATTN_BLOCK_IMG_URL)}" style="color: lightgrey;">Image Source</a>', unsafe_allow_html=True
    )
