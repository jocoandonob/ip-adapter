import streamlit as st
from dotenv import load_dotenv
from src.utils.template_loader import load_css, load_template
from src.components.text_to_image import render_text_to_image_tab
from src.components.image_to_image import render_image_to_image_tab
from src.config.constants import MODEL_CONFIGS

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="IP-Adapter Converter",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load CSS
load_css()

# Set theme colors and styling
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #2C1810;
        }
        
        /* Remove empty boxes above title */
        .main .block-container {
            padding-top: 0;
        }
        
        /* Title styling */
        .main .block-container h1 {
            font-size: 4rem !important;
            font-weight: 800 !important;
            text-align: center;
            color: #FFD700;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5),
                         4px 4px 8px rgba(0, 0, 0, 0.3),
                         6px 6px 12px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
            padding: 1rem;
            transition: all 0.3s ease;
        }
        
        .main .block-container h1:hover {
            transform: scale(1.02);
            text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.6),
                         6px 6px 12px rgba(0, 0, 0, 0.4),
                         9px 9px 18px rgba(0, 0, 0, 0.3);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            padding: 0.5rem;
            background-color: #1A0F0A;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3.5rem;
            white-space: pre-wrap;
            background-color: #2C1810;
            border-radius: 8px;
            gap: 1rem;
            padding: 0.5rem 1.5rem;
            color: #FFD700;
            font-weight: 600;
            transition: all 0.3s ease;
            border: 1px solid #4A2C1A;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #3D2315;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #4A2C1A;
            color: #FFD700;
            border: 1px solid #FFD700;
            box-shadow: 0 4px 12px rgba(255, 215, 0, 0.2);
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 2rem;
        }
        
        /* File uploader styling */
        .stFileUploader {
            background-color: #1A0F0A;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #4A2C1A;
        }
        
        .stFileUploader:hover {
            border-color: #FFD700;
        }
        
        /* Selectbox styling */
        .stSelectbox {
            background-color: #1A0F0A;
            border-radius: 8px;
            border: 1px solid #4A2C1A;
        }
        
        .stSelectbox:hover {
            border-color: #FFD700;
        }
        
        /* Text area styling */
        .stTextArea textarea {
            background-color: #1A0F0A;
            border: 1px solid #4A2C1A;
            color: #FFD700;
            border-radius: 8px;
        }
        
        .stTextArea textarea:focus {
            border-color: #FFD700;
            box-shadow: 0 0 0 1px #FFD700;
        }
        
        /* Slider styling */
        .stSlider {
            background-color: #1A0F0A;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #4A2C1A;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #4A2C1A;
            color: #FFD700;
            border: 1px solid #FFD700;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: #FFD700;
            color: #2C1810;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3);
        }
        
        /* Card styling */

        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #FFD700;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("IP-Adapter Converter")
st.markdown(load_template("cards").split("<!-- Description Card -->")[1], unsafe_allow_html=True)

# Style selection
selected_style = st.selectbox(
    "Select Style:",
    options=list(MODEL_CONFIGS.keys()),
    format_func=lambda x: x
)

# Tab selection
tab1, tab2 = st.tabs(["Text to Image", "Image to Image"])

# Render tabs
with tab1:
    render_text_to_image_tab(selected_style)

with tab2:
    render_image_to_image_tab(selected_style) 