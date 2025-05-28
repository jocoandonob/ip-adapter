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
    DEFAULT_GUIDANCE_SCALE
)

def render_text_to_image_tab():
    # Create two columns for input and output
    col1, col2 = st.columns([1, 1])

    with col1:
        # Input section
        st.markdown(load_template("cards").split("<!-- Input Card -->")[1].split("<!-- Parameters Card -->")[0], unsafe_allow_html=True)
        
        # Model selection
        selected_model = st.selectbox(
            "Select Style:",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: x
        )
        
        # Load model configuration
        model_config = MODEL_CONFIGS[selected_model]
        
        # Text input with a larger text area
        prompt = st.text_area(
            "Enter your prompt:",
            height=80,
            placeholder=f"Describe the image you want to generate... (e.g., '{model_config['default_prompt']}')",
            value=model_config["default_prompt"]
        )
        
        # IP-Adapter image upload if enabled
        ip_adapter_image = None
        if model_config.get("use_ip_adapter", False):
            uploaded_file = st.file_uploader("Upload reference image for IP-Adapter", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                ip_adapter_image = Image.open(uploaded_file)
                st.image(ip_adapter_image, caption="Reference Image", use_column_width=True)
        
        # Parameters section
        st.markdown(load_template("cards").split("<!-- Parameters Card -->")[1].split("<!-- Output Card -->")[0], unsafe_allow_html=True)
        
        num_inference_steps = st.slider("Number of inference steps", 20, 100, DEFAULT_STEPS)
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, DEFAULT_GUIDANCE_SCALE)
        
        # IP-Adapter scale if enabled
        ip_adapter_scale = None
        if model_config.get("use_ip_adapter", False):
            ip_adapter_scale = st.slider("IP-Adapter influence", 0.0, 1.0, model_config.get("ip_adapter_scale", 0.6))
        
        # Seed control
        seed = st.number_input("Seed (for reproducibility)", value=DEFAULT_SEED, step=1)
        
        # Generate button
        generate_button = st.button("🎨 Generate Image", type="primary")

    with col2:
        # Output section
        st.markdown(load_template("cards").split("<!-- Output Card -->")[1], unsafe_allow_html=True)
        
        # Placeholder for the generated image
        image_placeholder = st.empty()
        
        if generate_button:
            if prompt:
                try:
                    # Create loading animation
                    loading_container = st.empty()
                    with loading_container:
                        st.spinner("Loading model...")
                    
                    # Load model
                    pipe = load_model(selected_model)
                    
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
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
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
                    image_placeholder.image(image, caption=f"Generated Image using {selected_model} style", use_container_width=True)
                    
                    # Add download button
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Image",
                        data=buf.getvalue(),
                        file_name=f"{selected_model.lower()}_style_output.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    # Clear all loading states
                    loading_container.empty()
                    progress_bar.empty()
                    progress_text.empty()
                    time_text.empty()
                    st.error(f"Error generating image: {str(e)}")
            elif generate_button:
                st.warning("⚠️ Please enter a prompt first.") 