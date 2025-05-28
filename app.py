import streamlit as st
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
from PIL import Image
import io
from huggingface_hub import hf_hub_download
import os

# Set page config
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state for storing models
if 'text2img_pipeline' not in st.session_state:
    st.session_state.text2img_pipeline = None
if 'img2img_pipeline' not in st.session_state:
    st.session_state.img2img_pipeline = None

# Function to load models
@st.cache_resource
def load_models():
    # Download IP-Adapter model file
    ip_adapter_path = hf_hub_download(
        repo_id="h94/IP-Adapter",
        filename="ip-adapter_sdxl.bin",
        subfolder="sdxl_models"
    )
    
    # Download base model files
    base_model_path = hf_hub_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="model_index.json"
    )
    
    # Load text-to-image pipeline
    text2img_pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,  # Changed to float32 for CPU
        use_safetensors=True,
        local_files_only=False
    )
    
    # Load IP-Adapter weights
    text2img_pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
        local_files_only=False
    )
    text2img_pipeline.set_ip_adapter_scale(0.6)

    # Load image-to-image pipeline
    img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,  # Changed to float32 for CPU
        use_safetensors=True,
        local_files_only=False
    )
    
    # Load IP-Adapter weights
    img2img_pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
        local_files_only=False
    )
    img2img_pipeline.set_ip_adapter_scale(0.6)

    return text2img_pipeline, img2img_pipeline

# Main app
st.title("AI Image Generator")
st.write("Generate images using text prompts or transform existing images")
st.warning("⚠️ Running on CPU - Generation will be slower than GPU")

# Sidebar for model loading
with st.sidebar:
    st.header("Model Settings")
    if st.button("Load Models"):
        with st.spinner("Loading models..."):
            st.session_state.text2img_pipeline, st.session_state.img2img_pipeline = load_models()
        st.success("Models loaded successfully!")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Text to Image", "Image to Image"])

# Text to Image Tab
with tab1:
    st.header("Text to Image Generation")
    
    # Text input
    prompt = st.text_input("Enter your prompt", "a polar bear sitting in a chair drinking a milkshake")
    negative_prompt = st.text_input("Enter negative prompt", "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality")
    
    # Image upload for IP adapter
    ip_image = st.file_uploader("Upload reference image for IP adapter", type=["png", "jpg", "jpeg"])
    
    # Generation parameters
    col1, col2 = st.columns(2)
    with col1:
        num_inference_steps = st.slider("Number of inference steps", 20, 50, 30)  # Reduced max steps for CPU
    with col2:
        seed = st.number_input("Random seed", value=0)
    
    if st.button("Generate Image") and st.session_state.text2img_pipeline is not None:
        if ip_image is not None:
            # Convert uploaded file to PIL Image
            ip_image = Image.open(io.BytesIO(ip_image.read()))
            
            with st.spinner("Generating image... (This may take a few minutes on CPU)"):
                generator = torch.Generator(device="cpu").manual_seed(seed)
                images = st.session_state.text2img_pipeline(
                    prompt=prompt,
                    ip_adapter_image=ip_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images
                
                st.image(images[0], caption="Generated Image")
        else:
            st.warning("Please upload a reference image for IP adapter")

# Image to Image Tab
with tab2:
    st.header("Image to Image Transformation")
    
    # Image upload
    input_image = st.file_uploader("Upload input image", type=["png", "jpg", "jpeg"])
    ip_image = st.file_uploader("Upload IP adapter image", type=["png", "jpg", "jpeg"])
    
    # Generation parameters
    col1, col2 = st.columns(2)
    with col1:
        strength = st.slider("Transformation strength", 0.0, 1.0, 0.6)
    with col2:
        seed = st.number_input("Random seed", value=4)
    
    if st.button("Transform Image") and st.session_state.img2img_pipeline is not None:
        if input_image is not None and ip_image is not None:
            # Convert uploaded files to PIL Images
            input_image = Image.open(io.BytesIO(input_image.read()))
            ip_image = Image.open(io.BytesIO(ip_image.read()))
            
            with st.spinner("Transforming image... (This may take a few minutes on CPU)"):
                generator = torch.Generator(device="cpu").manual_seed(seed)
                images = st.session_state.img2img_pipeline(
                    prompt="best quality, high quality",
                    image=input_image,
                    ip_adapter_image=ip_image,
                    generator=generator,
                    strength=strength,
                ).images
                
                st.image(images[0], caption="Transformed Image")
        else:
            st.warning("Please upload both input and IP adapter images") 