import numpy as np
import streamlit as st
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from src.constants import DEMO_IMAGES_DIR, DEFAULT_PATCH_SIZE, TARGET_IMAGE_SIZE
from src.utils import img_to_patches, load_hf_model, load_labels, generate_attention_maps, visualize_attention_maps

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

st.text("Selected image:")
img_path = options_to_img[selection].as_posix()
st.image(img_path, width=300)

# display selected image
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

st.divider()
st.subheader("Image Patches")

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
    st.pyplot(fig, use_container_width=False)

# compute attention maps
model, feature_extractor = load_hf_model()
inputs = feature_extractor(images=img_raw, return_tensors="pt", do_resize=True, size=TARGET_IMAGE_SIZE)
labels = load_labels()

# get model outputs
with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True,
        output_hidden_states=True,
        interpolate_pos_encoding=False,
        return_dict=True
    )

logits = outputs.logits
predicted_class = logits.argmax(dim=1).item()
predicted_label = labels[predicted_class]

attentions = outputs.attentions
num_layers = len(attentions)
_, num_heads, num_patches, sequence_length = attentions[-1].shape

st.subheader("Model Parameters and Embedding Dimensions")
st.text(f"""
    Predicted label: {predicted_label.capitalize()}
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
    help="Consolidation method across heads in a MHA Layer"
)
with st.spinner("Generating Attention Maps"):
    attention_maps = generate_attention_maps(attentions, fusion_method.lower())
    st.pyplot(visualize_attention_maps(img_raw, attention_maps))

    st.markdown("**👓 Observation**: As the network grows deeper, visualizing raw attention is not sufficient and "
                "results in less reliable explanations.")

st.divider()



