import streamlit as st
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from src.constants import DEMO_IMAGES_DIR, DEFAULT_PATCH_SIZE, TARGET_IMAGE_SIZE
from src.utils import img_to_patches, load_vit_model, load_labels, generate_attention_maps, visualize_attention_maps, \
    compute_attention_rollout, get_attention_rollout_per_layer, visualize_attention_rollout

st.title("🤖 Vision Transformers - Explained")

st.text("Enjoy this short demonstration of Vision Transformers!")

st.header("From Image to Classification")

# display ViT GIF
st.video(
    './data/vit-gif.mp4',
    loop=True,
    autoplay=True,
    muted=True,
)
st.markdown(
    '<a href="https://research.google/blog/transformers-for-image-recognition-at-scale/" style="color: lightgrey;">GIF Source</a>',
    unsafe_allow_html=True
)

# display pages
st.divider()
st.header("🖼️ Visualize Patches")

st.text("""
    Select an object to visualize how Vision Transformers patchify an image.
""")

options_to_img = {
    "Cab": DEMO_IMAGES_DIR / 'yellow_cab.jpeg',
    "Airplane": DEMO_IMAGES_DIR / 'airplane.jpg',
    # TODO: add more demo images
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

# display patch sizes
available_patch_sizes = [i for i in range(2, img_raw.size[0], 2) if img_raw.size[0] % i == 0]
patch_size = st.select_slider(
    label="Select a patch size $P$",
    options=available_patch_sizes,
    value=DEFAULT_PATCH_SIZE
)

st.markdown("Number of image patches $N$:")
# display the number of image patches
st.latex(fr"N=HW/P^2={img_raw.size[0]}\cdot{img_raw.size[1]}/{patch_size}^2={int((img_raw.size[0]*img_raw.size[1]) / patch_size**2)}")

left_col, right_col = st.columns(2)

with left_col:
    # display selected image
    st.text("Selected image")
    st.image(img_path, use_container_width=True)

with right_col:
    st.text("Image Patches")

    with st.spinner('Splitting image into patches'):
        # split image into patches
        img_patches = img_to_patches(img, patch_size)
        # Visualize patches for the first image in the batch
        B, C, num_patches_y, num_patches_x, _, _ = img_patches.shape
        fig, axs = plt.subplots(num_patches_y, num_patches_x, figsize=(5, 5))

        for i in range(num_patches_y):
            for j in range(num_patches_x):
                # Extract the patch (first image in the batch)
                patch = img_patches[0, :, i, j].permute(1, 2, 0).numpy()  # Convert to [H, W, C] for visualization
                # normalize patch values to [0, 1]
                patch = (patch + 1) / 2
                axs[i, j].imshow(patch)
                axs[i, j].axis("off")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

st.divider()

# compute attention maps
model = load_vit_model(patch_size=patch_size)
labels = load_labels()

# get model outputs
with torch.no_grad():
    logits = model(img)

predicted_class = logits.argmax(dim=1).item()
predicted_label = labels[predicted_class]
prediction_proba = round(torch.max(F.softmax(logits, dim=1), dim=1).values.item(), 2)

# attentions = outputs.attentions
num_layers = len(model.blocks)
_, num_heads, num_patches, sequence_length = model.blocks[-1].attn.attn_map.shape

st.subheader("Model Parameters and Embedding Dimensions")
st.text(f"""
    Predicted label: {predicted_label.capitalize()}
    Prediction confidence: {prediction_proba}
    Number of Attention Layers: {num_layers}
    Number of Self-attention Heads: {num_heads}
    Number of Tokens: {num_patches - 1} + 1 CLS token
    Sequence Length: {sequence_length}
""")

st.subheader("Attention Maps per Layer")
st.markdown(r"Attention Weights = $\text{softmax}(\frac{\mathbf{QK}^T}{\sqrt{d}}) \in [0,1]$")

st.text("In order to capture the attention weights from a Multi-headed Attention Layer, the attention weights are "
        "consolidated across the heads.")

fusion_method = st.selectbox(
    label="Fusion Method",
    options=["Mean", "Min", "Max"],
    index=2,
    help="Fusion method across heads in a MHA Layer"
)
# extract attention maps for CLS token from each model block
attention_maps: list[torch.Tensor] = generate_attention_maps(model, method=fusion_method.lower())

with st.spinner("Compute Attention Rollout"):
    attention_rollout = compute_attention_rollout(attention_maps, grid_size=patch_size, image_size=img_raw.size)
    st.pyplot(visualize_attention_rollout(img_raw, attention_rollout, patch_size))
