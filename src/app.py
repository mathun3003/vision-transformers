import streamlit as st
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from src.constants import DEMO_IMAGES_DIR, DEFAULT_PATCH_SIZE, TARGET_IMAGE_SIZE, ATTN_BLOCK_IMG_URL, VIT_GIF_URL
from src.utils import img_to_patches, load_vit_model, load_labels, compute_attention_rollout, \
    visualize_attention_rollout

# TODO: set up streamlit config
# - full width for app
# - color theme

# load ImageNet labels
labels = load_labels()

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
    f'<a href="{str(VIT_GIF_URL)}" style="color: lightgrey;">GIF Source</a>',
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
    "Chihuahua": DEMO_IMAGES_DIR / 'chihuahua.jpg',
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

# display patch sizes
available_patch_sizes = [i for i in range(2, img_raw.size[0], 2) if img_raw.size[0] % i == 0]
patch_size = st.select_slider(
    label="Select a patch size $P$",
    options=available_patch_sizes,
    value=DEFAULT_PATCH_SIZE
)

st.markdown("Number of image patches $N$:")
# display the number of image patches
st.latex(
    fr"N=HW/P^2={img_raw.size[0]}\cdot{img_raw.size[1]}/{patch_size}^2={int((img_raw.size[0] * img_raw.size[1]) / patch_size ** 2)}")

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
                # Convert to [H, W, C] for visualization
                patch = img_patches[0, :, i, j].permute(1, 2, 0).numpy()
                # normalize patch values to [0, 1]
                patch = (patch + 1) / 2
                axs[i, j].imshow(patch)
                axs[i, j].axis("off")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

st.divider()

with st.spinner("Performing model prediction"):
    model, feature_extractor = load_vit_model(patch_size=patch_size, image_size=img_raw.size[0])

# get model outputs
inputs = feature_extractor(images=img_raw, return_tensors="pt", do_resize=True, size=TARGET_IMAGE_SIZE)
outputs = model(**inputs)

# get prediction
logits = outputs.logits
predicted_class = logits.argmax(dim=1).item()
predicted_label = labels[predicted_class]
prediction_proba = round(torch.max(F.softmax(logits, dim=1), dim=1).values.item(), 2)

# NOTE: Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads
attentions = outputs.attentions

num_layers = len(outputs.attentions)
_, num_heads, num_patches, sequence_length = outputs.attentions[-1].shape

st.subheader("Model Parameters and Embedding Dimensions")
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

st.subheader("Attention Rollout")
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

# TODO: add skip connection computation

fusion_method = st.selectbox(
    label="Fusion Method",
    options=["Mean", "Min", "Max"],
    help="Fusion method across heads in a MHA Layer"
)
left_col, right_col = st.columns(spec=[.7, .3], vertical_alignment="center")
with left_col:
    with st.spinner("Computing Attention Rollout"):
        attention_rollout = compute_attention_rollout(attentions, fusion_method=fusion_method.lower())
        st.pyplot(visualize_attention_rollout(img_raw, attention_rollout, patch_size, num_layers))
with right_col:
    st.image(str(ATTN_BLOCK_IMG_URL), caption="Encoder Layer in ViT")
    st.markdown(
        f'<a href="{str(ATTN_BLOCK_IMG_URL)}" style="color: lightgrey;">Image Source</a>',
        unsafe_allow_html=True
    )
