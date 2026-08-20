"""
This is an interactive demo of the complete 4-stream model. Upload a document
image, see the original alongside the predicted tampering mask, plus a plain-language
verdict.

It includes connected-region filtering in the verdict logic: requires a real, minimum
connected cluster of flagged pixels before calling something "forged," not just any
single, isolated pixel spike, this came from testing on genuinely authentic images and
seeing false positives driven by small, isolated spikes, often around text.
"""

import streamlit as st
import torch
import numpy as np
import cv2
import io
import tempfile
import scipy.ndimage as ndi
from PIL import Image

from streams.model_streams import RGBStream, ELAStream
from streams.dct_stream import DCTStream
from streams.noise_stream import NoiseStream
from streams.gated_fusion_head import GatedFusionHead

st.set_page_config(page_title="Document Forgery Detection", layout="wide")
st.title("Document Forgery Detection")
st.markdown(
    "A 4-stream deep learning model (RGB + Error Level Analysis + DCT + "
    "Noise-Fingerprint) that detects splicing, text tampering, and "
    "AI-generated document forgery, at the pixel level. "
    "Upload a document image below to see it in action."
)

CHECKPOINT_PATH = 'results/checkpoint_stage2_final_4stream.pt'
RESIZE_TO = 512
DEFAULT_THRESHOLD = 0.60
DEFAULT_MIN_REGION = 100


@st.cache_resource
def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    rgb_model = RGBStream()
    ela_model = ELAStream()
    dct_model = DCTStream()
    noise_model = NoiseStream()
    head = GatedFusionHead()
    rgb_model.load_state_dict(ckpt['rgb_model_state'])
    ela_model.load_state_dict(ckpt['ela_model_state'])
    dct_model.load_state_dict(ckpt['dct_model_state'])
    noise_model.load_state_dict(ckpt['noise_model_state'])
    head.load_state_dict(ckpt['head_state'])
    rgb_model.eval(); ela_model.eval(); dct_model.eval(); noise_model.eval(); head.eval()
    return rgb_model, ela_model, dct_model, noise_model, head, ckpt.get('epoch', '?')


def compute_ela(pil_image, quality=90):
    buffer = io.BytesIO()
    pil_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    original_np = np.array(pil_image.convert("RGB")).astype(np.int16)
    resaved_np = np.array(resaved.convert("RGB")).astype(np.int16)
    return np.abs(original_np - resaved_np).astype(np.uint8)


def compute_dct_volume(pil_image, T=20, quality=90):
    import jpegio
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
        pil_image.convert('RGB').save(tmp.name, 'JPEG', quality=quality)
        jpg = jpegio.read(tmp.name)
        dct = jpg.coef_arrays[0].copy()
    clipped = np.clip(dct, -T, T)
    clipped = cv2.resize(clipped, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_NEAREST)
    H, W = clipped.shape
    num_layers = 2 * T + 1
    volume = np.zeros((num_layers, H, W), dtype=np.float32)
    for i, value in enumerate(range(-T, T + 1)):
        volume[i] = (clipped == value).astype(np.float32)
    return volume


def run_prediction(pil_image, rgb_model, ela_model, dct_model, noise_model, head):
    image = pil_image.convert('RGB').resize((RESIZE_TO, RESIZE_TO))
    ela = compute_ela(image)
    ela_resized = cv2.resize(ela, (RESIZE_TO, RESIZE_TO))
    dct_volume = compute_dct_volume(image)

    rgb_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ela_tensor = torch.from_numpy(ela_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dct_tensor = torch.from_numpy(dct_volume).unsqueeze(0)

    with torch.no_grad():
        _, rgb_feat = rgb_model(rgb_tensor)
        _, ela_feat = ela_model(ela_tensor)
        _, dct_feat = dct_model(dct_tensor)
        _, noise_feat = noise_model(rgb_tensor)
        logits = head([rgb_feat, ela_feat, dct_feat, noise_feat])
        probs = torch.sigmoid(logits).squeeze().numpy()

    return np.array(image), probs


with st.sidebar:
    st.header("About this model")
    st.markdown(
        "- **4 streams**: RGB, ELA, DCT, Noise-Fingerprint\n"
        "- **Trained on**: DocTamper (splicing/generation) + AIForge-Doc (AI-inpainting)\n"
        "- **Real evaluation results**:\n"
        "  - DocTamper test: F1 = 0.93\n"
        "  - AIForge-Doc test: F1 = 0.92\n"
        "  - FCD cross-domain test: F1 = 0.91\n"
    )
    threshold = st.slider(
        "Detection sensitivity (threshold)",
        min_value=0.05, max_value=0.95, value=DEFAULT_THRESHOLD, step=0.05,
    )
    min_region = st.slider(
        "Minimum region size (pixels)",
        min_value=0, max_value=1000, value=DEFAULT_MIN_REGION, step=25,
        help="Requires a real, connected cluster of this many flagged pixels "
             "before calling something forged, filters out isolated, noisy "
             "single-pixel spikes (e.g. from text edges)."
    )

uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    with st.spinner("Loading model..."):
        rgb_model, ela_model, dct_model, noise_model, head, epoch = load_model()

    with st.spinner("Analyzing document..."):
        original_np, probs = run_prediction(pil_image, rgb_model, ela_model, dct_model, noise_model, head)

    max_confidence = float(probs.max())
    mean_confidence = float(probs.mean())

    binary_mask = (probs > threshold).astype(np.uint8)
    labeled, num_features = ndi.label(binary_mask)
    has_real_region = False
    largest_region = 0
    if num_features > 0:
        region_sizes = ndi.sum(binary_mask, labeled, range(1, num_features + 1))
        largest_region = int(region_sizes.max())
        has_real_region = largest_region >= min_region

    simple_verdict = "LIKELY FORGED" if max_confidence > threshold else "LIKELY AUTHENTIC"
    refined_verdict = "LIKELY FORGED" if has_real_region else "LIKELY AUTHENTIC"
    verdict_color = "red" if refined_verdict == "LIKELY FORGED" else "green"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Document")
        st.image(original_np, use_container_width=True)
    with col2:
        st.subheader("Predicted Tampering Mask")
        heatmap = (probs * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        st.image(heatmap_color, use_container_width=True, caption="Brighter/warmer = more suspicious")
    with col3:
        st.subheader("Overlay")
        overlay = original_np.copy().astype(np.float32)
        mask_3ch = np.stack([probs, np.zeros_like(probs), np.zeros_like(probs)], axis=-1)
        overlay = overlay * 0.6 + mask_3ch * 255 * 0.4
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        st.image(overlay, use_container_width=True, caption="Red = suspicious regions")

    st.markdown("---")
    st.markdown(f"### Refined Verdict: :{verdict_color}[{refined_verdict}]")
    st.caption(f"(Simple max-pixel check alone would have said: {simple_verdict})")
    st.markdown(f"**Highest confidence anywhere in image:** {max_confidence:.1%}")
    st.markdown(f"**Average confidence across image:** {mean_confidence:.1%}")
    st.markdown(f"**Largest connected suspicious region:** {largest_region} pixels")
    st.caption(
        f"Model checkpoint: epoch {epoch}. This is a research prototype, "
        "results should be treated as a diagnostic signal, not a certified determination."
    )
else:
    st.info("Upload a document image above to see the model's real-time prediction.")