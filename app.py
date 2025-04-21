import streamlit as st
import requests
import base64
import os
from tempfile import NamedTemporaryFile
from main import generate_image_with_controlnet_v2

API_URL = "http://127.0.0.1:8000/generate"

st.set_page_config(page_title="Stable Diffusion UI", layout="wide")
st.title("🎨 AI Image Generator (Stable Diffusion + ControlNet)")

tab1, tab2 = st.tabs(["🧠 Basic Txt2Img", "📷 ControlNet"])

with tab1:
    st.header("🧠 Basic Generation")
    prompt = st.text_input("Prompt", "")
    negative_prompt = st.text_input("Negative Prompt", "(worst quality, low quality, jpeg artifacts)")
    width = st.slider("Width", 256, 1024, 512, step=64)
    height = st.slider("Height", 256, 1024, 512, step=64)
    seed = st.number_input("Seed (Leave 0 for random)", value=0)
    return_base64 = st.checkbox("Return Base64")

    if st.button("Generate Image"):
        with st.spinner("Generating..."):
            response = requests.post(API_URL, json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "seed": seed if seed != 0 else None,
                "return_base64": return_base64
            })

            if response.ok:
                r = response.json()
                st.success(r["message"])
                st.image(r["file"])
                if return_base64:
                    st.text_area("Base64 Image", r["base64_image"], height=200)
            else:
                st.error("Image generation failed.")
                st.json(response.json())

with tab2:
    st.header("📷 ControlNet Generation")
    control_prompt = st.text_input("Prompt", key="control_prompt")
    control_negative = st.text_input("Negative Prompt", "(worst quality, low quality, jpeg artifacts)", key="control_negative")

    uploaded_image = st.file_uploader("Upload base image", type=["png", "jpg", "jpeg"])

    width2 = st.slider("Width", 256, 1024, 512, key="w2")
    height2 = st.slider("Height", 256, 1024, 512, key="h2")
    seed2 = st.number_input("Seed (Leave -1 for random)", value=-1, key="s2")

    if st.button("Generate with ControlNet"):
        if not uploaded_image:
            st.warning("Please upload a base image.")
        else:
            with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(uploaded_image.read())
                temp_path = temp_file.name

            with st.spinner("Generating with ControlNet..."):
                try:
                    result = generate_image_with_controlnet_v2(
                        image_path=temp_path,
                        prompt=control_prompt,
                        negative_prompt=control_negative,
                        width=width2,
                        height=height2,
                        seed=seed2
                    )
                    st.success(result["message"])
                    st.image(result["image_path"])
                    st.text_area("Base64 Image", result["base64_image"], height=200)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
