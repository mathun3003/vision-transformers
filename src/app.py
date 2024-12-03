import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from src.constants import DEMO_IMAGES_DIR, PATCH_SIZE, TARGET_IMAGE_SIZE, ATTN_BLOCK_IMG_URL, VIT_GIF_URL
from src.utils import img_to_patches, load_vit_model, load_labels, compute_attention_rollout, \
    visualize_attention_rollout, visualize_qkv, compute_scaled_qkv

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
st.markdown(
    f'<a href="{str(VIT_GIF_URL)}" style="color: lightgrey;">GIF Source</a>',
    unsafe_allow_html=True
)

# display pages
st.divider()
st.subheader("🖼️ Visualize Patches")

st.text("""
    Select an object to visualize how Vision Transformers patchify an image.
""")

options_to_img = {
    "Cab": DEMO_IMAGES_DIR / 'yellow_cab.jpeg',
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

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
# apply torch transformation, remove batch dimension
img = transform(img_raw).unsqueeze(0)

st.markdown("Number of image patches $N$:")
# display the number of image patches
st.latex(
    fr"N=HW/P^2={img_raw.size[0]}\cdot{img_raw.size[1]}/{PATCH_SIZE}^2={int((img_raw.size[0] * img_raw.size[1]) / PATCH_SIZE ** 2)}")

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
        B, C, num_patches_y, num_patches_x, _, _ = img_patches.shape
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


def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
    global queries, keys, values
    queries = module.query(input[0])
    keys = module.key(input[0])
    values = module.value(input[0])


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

num_layers = len(outputs.attentions)
_, num_heads, num_patches, sequence_length = outputs.attentions[-1].shape

st.subheader("🎛️ Model Parameters and Embedding Dimensions")
st.markdown(f"""
    **Model Prediction**  
    Predicted label: {predicted_label.capitalize()}  
    Prediction confidence: {prediction_proba}  
    **Internal Model Parameters**  
    Number of Attention Layers: {num_layers}  
    Number of Self-attention Heads: {num_heads}  
    Number of Tokens: {num_patches - 1} + 1 CLS token  
    Sequence Length: {sequence_length}
""")

st.divider()

st.subheader("🔍 Queries, Keys, and Values")
st.text("Since the image is split into patches, the queries, keys, and values represent the transformed and embedded "
        "input image inside the transformer. One can visualize the query, key, and value image for a given layer, "
        "head, and embedding dimension.")
st.markdown(r"For each self-attention layer, the input embedding $\mathbf{z_i} \in \mathbb{R}^{N \times d}$ goes "
            r"through three linear projection layers with the weight matrices "
            r"$W^q \in \mathbb{R}^{d \times d}, \; W^k \in \mathbb{R}^{d \times d},\; W^v \in \mathbb{R}^{d \times d}$ "
            r"to compute the queries, keys, and values. ")
st.latex(
    r"\mathbf{Q}=\mathbf{z_i}W^q,\; \mathbf{K}=\mathbf{z_i}W^k,\; \mathbf{V}=\mathbf{z_i}W^v \in \mathbb{R}^{N \times d}")
st.markdown(r"Queries and keys are used to generate the so-called attention weights in self-attention $\mathbf{QK}^T$. "
            r"Then, the self-attention is computed as follows:")
st.latex(
    r"\text{SA}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\text{softmax}(\frac{\mathbf{QK}^T}{\sqrt{d}})\mathbf{V} \in \mathbb{R}^{N\times d}")

selected_channels = [21, 22, 26, 42]
with st.spinner("Computing Queries, Keys, and Values"):
    embedding_dim: int = queries.shape[-1]
    scaled_qkv = {
        channel: compute_scaled_qkv(img_raw.size, queries, keys, values, channel, embedding_dim)
        for channel in selected_channels
    }
    st.pyplot(visualize_qkv(img_raw, scaled_qkv, selected_channels, layer_idx))

st.divider()

st.subheader("🔁 Attention Rollout")
st.text("In order to see how the attention flows through the network, multiplying the attention weights from each "
        "attention block recursively at each layer results in the so-called attention rollout. In a Multi-head "
        "Attention Layer, the self-attentions are fused together to a single attention weight matrix using either the "
        "maximum, minimum, or mean across the heads' attentions")
st.markdown(r"For $L$ layers, the attention rollout $\tilde{A}^{(l)}$ at layer $l \in \{1,…,L\}$ is recursively "
            r"defined as:")
st.latex(r"""
    \tilde{A}^{(l)}=
    \begin{cases}
    A^{(l)}\tilde{A}^{(l-1)} & \text{if } l > 1 \\
    A^{(l)} & \text{if } l=1
    \end{cases}
""")
st.markdown(r"Where $A^{(l)}$ is the raw attention of layer $l$. To incorporate the skip connections around the "
            r"Multi-head Attention Layer, $A$ is computed as the average of the input activations and the attention "
            r"weights $W_{attn}$")
st.latex(r"A^{(l)}=0.5 W_{attn}^{(l)} + 0.5I^{(l)}")


fusion_method = st.selectbox(
    label="Fusion Method",
    options=["Mean", "Min", "Max"],
    help="Fusion method across heads in a MHA Layer"
)
left_col, right_col = st.columns(spec=[.7, .3], vertical_alignment="center")
with left_col:
    with st.spinner("Computing Attention Rollout"):
        attention_rollout = compute_attention_rollout(attentions, fusion_method=fusion_method.lower())
        st.pyplot(visualize_attention_rollout(img_raw, attention_rollout, PATCH_SIZE, num_layers))
with right_col:
    st.image(str(ATTN_BLOCK_IMG_URL), caption="Encoder Layer in ViT")
    st.markdown(
        f'<a href="{str(ATTN_BLOCK_IMG_URL)}" style="color: lightgrey;">Image Source</a>',
        unsafe_allow_html=True
    )
