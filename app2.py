import streamlit as st
from gptgen import gptgen
from text2speech import speak
import time
from motionmain import main
from PIL import Image
import yaml



# Set the page title
st.set_page_config(page_title="EchoSign", page_icon="🔊")

def load_config():
    """
    Load the configuration from a YAML file.

    Returns:
        dict: The loaded configuration.
    """
    try:
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        st.error(f"Error loading configuration file: {e}")
        return None


config = load_config()
# Load and display the logo image with circular styling
# 
# logo = Image.open("image/logo.png")

#try:
#    st.sidebar.image(config['paths']['logo_path'], use_container_width=True)
#except Exception as e:
#    st.error(f"Error loading logo image: {e}")
# st.markdown(
#     """
#     <style>
#     .logo-container {
#         display: flex;
#         justify-content: center;
#         margin-top: 20px;
#     }
#     .logo-img {
#         width: 150px;  /* Adjust the size as needed */
#         height: 150px;
#         border-radius: 50%;  /* Makes the image circular */
#         object-fit: cover;  /* Ensures the image covers the circular area */
#         border: 3px solid #FFFFFF;  /* Optional: Add a border around the circle */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# st.markdown('<div class="logo-container"><img src="image/logo.png" class="logo-img"/></div>', unsafe_allow_html=True)

# Initialize camera state in session state
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# Apply custom CSS styling for a black theme with Instagram-themed gradient accents
st.markdown(
    """
    <style>
    /* Full-page gradient background for Instagram-like theme */
    body {
        background: linear-gradient(135deg, #833AB4, #FD1D1D, #FCB045);
        font-family: 'Helvetica', sans-serif;
        color: #FFFFFF;
    }

    /* Dark overlay to make content stand out */
    .overlay {
        background-color: rgba(0, 0, 0, 0.85);
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
    }

    /* Main content styling with Instagram theme */
    .main-content {
        position: relative;
        z-index: 1;
        padding-top: 50px;
    }

    /* Main title with Instagram gradient text */
    .title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #833AB4, #FD1D1D, #FCB045);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 50px;
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 1.5rem;
        text-align: center;
        color: #b0b0b0;
        margin-top: 10px;
    }

    /* Input box styling */
    .email-input {
        width: 80%;
        margin: 0 auto;
        display: block;
        padding: 10px;
        font-size: 1.2rem;
        border-radius: 8px;
        border: 1px solid #444;
        background-color: #222;
        color: #FFFFFF;
        text-align: center;
    }

    /* Button styling */
    .join-button, .camera-button, .speaker-button {
        display: inline-block;
        margin: 20px 10px;
        padding: 15px 30px;
        background: linear-gradient(135deg, #833AB4, #FD1D1D, #FCB045);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .join-button:hover, .camera-button:hover, .speaker-button:hover {
        background: #FCB045;
    }

    /* Speaker button with emoji icon */
    .speaker-button {
        font-size: 1.5rem;
        padding: 10px 20px;
    }

    /* Center content */
    .center-content {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add overlay for black background with Instagram theme accents
st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)

# Title for the app
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown('<h1 class="title">SignEcho</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="title">Bridge the Communication Gap</h2>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Let\'s talk !!! </p>', unsafe_allow_html=True)




# Define the gen function
def gen():
    return gptgen()

# Placeholder for generated text
generated_text = ""

# Join waitlist button functionality
if st.button("Start Interpreting", key="join_button"):
    generated_text = gen()
    st.success(f"{generated_text}")

# Speaker button for text-to-speech if generated text exists
if st.button("🔊 Speak", key="speaker_button", help="Click to hear the generated response"):
    speak()

# Side-by-side Start and Stop buttons for Camera control
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Camera", key="start_camera"):
        st.session_state.camera_on = True

with col2:
    if st.button("Stop Camera", key="stop_camera"):
        st.session_state.camera_on = False

# Run camera only if it is on
if st.session_state.camera_on:
    st.info("Camera is on.")
    main()
    time.sleep(0.1)  # Short sleep to simulate continuous checking
else:
    st.info("Camera is off.")

st.markdown('</div>', unsafe_allow_html=True)  # End main content