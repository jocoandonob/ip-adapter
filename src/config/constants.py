"""
Constants and configuration values for the Stable Diffusion Image Generator.
"""

# Model configurations for different styles
MODEL_CONFIGS = {

    "TextToImage": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "lora_path": None,
        "default_prompt": "a polar bear sitting in a chair drinking a milkshake",
        "use_safetensors": True,
        "is_sdxl": True,
        "pipeline": "sdxl",
        "ip_adapter": {
            "model_path": "h94/IP-Adapter",
            "subfolder": "sdxl_models",
            "weight_name": "ip-adapter_sdxl.bin",
            "scale": 0.6
        },
        "negative_prompt": "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        "num_inference_steps": 100
    }
}

# File paths
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"
CSS_DIR = "css"
MODELS_DIR = "models"

# UI Constants
DEFAULT_SEED = 123
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_STRENGTH = 0.75

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg"] 