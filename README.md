# AI Image Generator

A Streamlit application that combines text-to-image and image-to-image generation capabilities using Stable Diffusion XL and IP-Adapter. This application allows users to generate and transform images using both text prompts and reference images.

## Features

- **Text to Image Generation**
  - Generate images from text prompts
  - Use reference images to guide the generation style
  - Customizable negative prompts
  - Adjustable generation parameters

- **Image to Image Transformation**
  - Transform existing images using reference styles
  - Adjustable transformation strength
  - Maintain image composition while applying new styles

- **User Interface**
  - Clean and intuitive Streamlit interface
  - Separate tabs for different functionalities
  - Real-time parameter adjustments
  - Progress indicators and error handling

## Requirements

- Python 3.8 or higher
- Sufficient RAM (8GB minimum recommended)
- Internet connection for model downloads
- Disk space for model storage (approximately 10GB)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Click "Load Models" in the sidebar to initialize the AI models
   - First-time loading will download the models (may take several minutes)
   - Models will be cached for subsequent runs

4. Choose your desired functionality:

   ### Text to Image
   - Enter your text prompt
   - Upload a reference image for style guidance
   - Adjust generation parameters:
     - Number of inference steps (20-50)
     - Random seed for reproducibility
   - Click "Generate Image"

   ### Image to Image
   - Upload your input image
   - Upload a reference image for style guidance
   - Adjust transformation parameters:
     - Transformation strength (0.0-1.0)
     - Random seed for reproducibility
   - Click "Transform Image"

## Performance Notes

- The application runs on CPU by default
- Image generation may take several minutes depending on your hardware
- For faster generation:
  - Use fewer inference steps (20-30)
  - Reduce image resolution if needed
  - Close other resource-intensive applications

## Troubleshooting

1. **Model Loading Issues**
   - Ensure stable internet connection
   - Check available disk space
   - Clear cache if needed: `rm -rf .cache/huggingface`

2. **Memory Issues**
   - Reduce batch size
   - Use fewer inference steps
   - Close other applications

3. **Generation Quality**
   - Adjust the IP-Adapter scale (default: 0.6)
   - Modify negative prompts
   - Try different random seeds

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [Streamlit](https://streamlit.io/)
- [Hugging Face](https://huggingface.co/) 