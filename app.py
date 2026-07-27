"""
app.py
------
Simple Streamlit web app for the nucleus/cell segmentation model.

Upload a microscopy image -> the app runs it through the trained U-Net
and shows: original image, predicted mask, and an overlay, plus basic
stats (estimated foreground coverage).

Run locally:
    streamlit run app.py

Requires a trained checkpoint at outputs/best_model.pt (see README /
src/train.py to produce one).
"""

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Make sure we can import the model code from src/
sys.path.append(str(Path(__file__).parent / "src"))
from model import UNet  # noqa: E402


CHECKPOINT_PATH = Path(__file__).parent / "outputs" / "best_model.pt"


@st.cache_resource
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet(in_channels=3, out_channels=1, base_ch=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["img_size"], device


def preprocess(image: Image.Image, img_size: int):
    image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor, image


def predict(model, device, tensor, threshold: float):
    with torch.no_grad():
        logits = model(tensor.to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    return prob, mask


def make_overlay(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.45):
    base = np.array(image).astype(np.float32)
    overlay = base.copy()
    color_arr = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    overlay[m] = (1 - alpha) * base[m] + alpha * color_arr
    return Image.fromarray(overlay.astype(np.uint8))


def main():
    st.set_page_config(page_title="Cell Segmentation", layout="centered")
    st.title("🔬 Cell / Nucleus Segmentation")
    st.write(
        "Upload a microscopy image and this U-Net model (trained on the "
        "Kaggle Data Science Bowl 2018 dataset) will highlight the "
        "detected nuclei/cells."
    )

    if not CHECKPOINT_PATH.exists():
        st.error(
            f"No trained checkpoint found at `{CHECKPOINT_PATH}`. "
            "Train a model first with `python src/train.py`, or place "
            "your `best_model.pt` in the `outputs/` folder."
        )
        st.stop()

    model, img_size, device = load_model(str(CHECKPOINT_PATH))
    st.caption(f"Model loaded on **{device}** · input size {img_size}x{img_size}")

    threshold = st.sidebar.slider("Segmentation threshold", 0.05, 0.95, 0.5, 0.05)

    uploaded_file = st.file_uploader(
        "Upload a microscopy image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        tensor, resized_image = preprocess(image, img_size)
        prob, mask = predict(model, device, tensor, threshold)
        overlay = make_overlay(resized_image, mask)

        coverage_pct = 100.0 * mask.sum() / mask.size

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(resized_image, caption="Input (resized)", use_container_width=True)
        with col2:
            st.image(mask * 255, caption="Predicted mask", use_container_width=True)
        with col3:
            st.image(overlay, caption="Overlay", use_container_width=True)

        st.metric("Estimated foreground (cell) coverage", f"{coverage_pct:.1f}%")

        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        st.download_button(
            "Download predicted mask (PNG)",
            data=buf.getvalue(),
            file_name="predicted_mask.png",
            mime="image/png",
        )
    else:
        st.info("Upload an image above to get started.")


if __name__ == "__main__":
    main()
