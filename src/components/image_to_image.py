import streamlit as st
import torch
import io
import time
from PIL import Image
from src.pipelines.model_loader import load_model
from src.utils.template_loader import load_template
from src.config.constants import (
    MODEL_CONFIGS,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_STRENGTH,
    SUPPORTED_IMAGE_FORMATS
)

def render_image_to_image_tab():
    # Create two columns for input and output
    col1, col2 = st.columns([1, 1])

    with col1:
        # Input section
        st.markdown(load_template("cards").split("<!-- Input Card -->")[1].split("<!-- Parameters Card -->")[0], unsafe_allow_html=True)
        
        # Model selection
        selected_model = st.selectbox(
            "Select Style (Image to Image):",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: x,
            key="img2img_model"
        )
        
        # Load model configuration
        model_config = MODEL_CONFIGS[selected_model]
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an image to transform:", type=SUPPORTED_IMAGE_FORMATS)
        
        if uploaded_file is not None:
            # Display the uploaded image
            init_image = Image.open(uploaded_file)
            st.image(init_image, caption="Original Image", use_container_width=True)
        
        # IP-Adapter image upload if enabled
        ip_adapter_image = None
        if model_config.get("use_ip_adapter", False):
            ip_uploaded_file = st.file_uploader("Upload reference image for IP-Adapter", type=SUPPORTED_IMAGE_FORMATS, key="img2img_ip_adapter")
            if ip_uploaded_file is not None:
                ip_adapter_image = Image.open(ip_uploaded_file)
                st.image(ip_adapter_image, caption="Reference Image", use_column_width=True)
        
        # Text input with a larger text area
        prompt = st.text_area(
            "Enter your prompt:",
            height=80,
            placeholder=f"Describe how you want to transform the image... (e.g., '{model_config['default_prompt']}')",
            value=model_config["default_prompt"],
            key="img2img_prompt"
        )
        
        # Parameters section
        st.markdown(load_template("cards").split("<!-- Parameters Card -->")[1].split("<!-- Output Card -->")[0], unsafe_allow_html=True)
        
        # Image size controls
        col_width, col_height = st.columns(2)
        with col_width:
            width = st.number_input("Width", min_value=256, max_value=1024, value=512, step=64, key="img2img_width")
        with col_height:
            height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=64, key="img2img_height")
        
        num_inference_steps = st.slider("Number of inference steps", 20, 100, DEFAULT_STEPS, key="img2img_steps")
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, DEFAULT_GUIDANCE_SCALE, key="img2img_guidance")
        strength = st.slider("Transformation strength", 0.0, 1.0, DEFAULT_STRENGTH, key="img2img_strength")
        
        # IP-Adapter scale if enabled
        ip_adapter_scale = None
        if model_config.get("use_ip_adapter", False):
            ip_adapter_scale = st.slider("IP-Adapter influence", 0.0, 1.0, model_config.get("ip_adapter_scale", 0.6), key="img2img_ip_scale")
        
        # Seed control
        seed = st.number_input("Seed (for reproducibility)", value=DEFAULT_SEED, step=1, key="img2img_seed")
        
        # Generate button
        generate_button = st.button("🎨 Transform Image", type="primary", key="img2img_generate")

    with col2:
        # Output section
        st.markdown(load_template("cards").split("<!-- Output Card -->")[1], unsafe_allow_html=True)
        
        # Placeholder for the generated image
        image_placeholder = st.empty()
        
        if generate_button and prompt and uploaded_file is not None:
            try:
                # Create loading animation
                loading_container = st.empty()
                with loading_container:
                    st.spinner("Loading model...")
                
                # Load model
                pipe = load_model(selected_model, img2img=True)
                
                # Create progress bar
                progress_bar = st.empty()
                progress_text = st.empty()
                time_text = st.empty()
                
                # Progress callback
                def progress_callback(step, timestep, latents):
                    progress = step / num_inference_steps
                    progress_bar.progress(progress)
                    progress_text.text(f"Step {step}/{num_inference_steps}")
                    time_text.text(f"Timestep: {timestep:.2f}")
                
                # Set up generator for reproducibility
                generator = torch.Generator(device="cpu").manual_seed(seed)
                
                # Prepare generation parameters
                gen_params = {
                    "prompt": prompt,
                    "image": init_image,
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "generator": generator,
                    "callback": progress_callback,
                    "callback_steps": 1
                }
                
                # Add IP-Adapter parameters if enabled
                if model_config.get("use_ip_adapter", False) and ip_adapter_image is not None:
                    gen_params["ip_adapter_image"] = ip_adapter_image
                    if ip_adapter_scale is not None:
                        pipe.set_ip_adapter_scale(ip_adapter_scale)
                
                # Generate image with progress callback
                image = pipe(**gen_params).images[0]
                
                # Clear loading animation and progress
                loading_container.empty()
                progress_bar.empty()
                progress_text.empty()
                time_text.empty()
                
                # Display image
                image_placeholder.image(image, caption=f"Transformed Image using {selected_model} style", use_container_width=True)
                
                # Add download button
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Image",
                    data=buf.getvalue(),
                    file_name=f"{selected_model.lower()}_style_transformed.png",
                    mime="image/png"
                )
                
            except Exception as e:
                # Clear all loading states
                loading_container.empty()
                progress_bar.empty()
                progress_text.empty()
                time_text.empty()
                st.error(f"Error transforming image: {str(e)}")
        elif generate_button:
            if not uploaded_file:
                st.warning("⚠️ Please upload an image first.")
            elif not prompt:
                st.warning("⚠️ Please enter a prompt first.") 