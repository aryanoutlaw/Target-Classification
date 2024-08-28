import streamlit as st
import torch
from PIL import Image
import os
from utils import TinyVGG, get_prediction
import time
import base64

# Function to load the model
def load_model():
    model = TinyVGG(in_channels=3, hidden_units=16, output_shape=6)  # Adjust these parameters as needed
    model.load_state_dict(torch.load("tinyvgg.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to load the custom CSS
def load_css():
    with open("static/styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to display the radar gif
def display_gif(placeholder, gif_path):
    file_ = open(gif_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    placeholder.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="Radar gif" width="300px" style="display: block; margin-left: auto; margin-right: auto;">',
        unsafe_allow_html=True,
    )

# Function to display the footer
def footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        .footer a {
            color: #f1c40f;
            text-decoration: none;
        }
        </style>
        <div class="footer">
            Developed with ❤️ by Team BruteForce | <a href="https://github.com/aryanoutlaw/Target-Classification">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main function
def main():
    st.set_page_config(page_title="Micro-Doppler Target Classification", page_icon="🎯", layout="centered")
    load_css()

    st.markdown("<h1 class='main-title'>🎯 Micro-Doppler Target Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Classify objects using micro-Doppler radar imagery. Upload or select an image to get started!</p>", unsafe_allow_html=True)
    
    model = load_model()
    with st.expander("How It Works"):
            st.write("""
            1. **Image Selection**: Choose a sample image or upload your own micro-Doppler image.
            2. **Preprocessing**: The image is resized and normalized to match the model's input requirements.
            3. **Model Prediction**: The TinyVGG model processes the image and outputs classification probabilities.
            4. **Result Display**: The highest probability class is shown as the prediction.
            """)

    # Sample Images Section
    st.subheader("Sample Images")
    sample_dir = "sample_images"
    sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_sample = st.selectbox("Select a sample image:", [""] + sample_images)
    
    if selected_sample:
        image_path = os.path.join(sample_dir, selected_sample)
        st.markdown("<div class='sample-image'>", unsafe_allow_html=True)
        st.image(image_path, caption='Selected Sample', width=300, use_column_width='auto')
        st.markdown("</div>", unsafe_allow_html=True)
        with open(image_path, "rb") as file:
            btn = st.download_button(
                label="Use this sample image",
                data=file,
                file_name=selected_sample,
                mime="image/png"
            )

    # Upload Image Section
    st.subheader("Upload Your Own Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown("<div class='uploaded-image'>", unsafe_allow_html=True)
        st.image(image, caption='Uploaded Image', width=300, use_column_width='auto')
        
    # Prediction Section
    if uploaded_file is not None or selected_sample:
        gif_placeholder = st.empty()  

        if st.button("Predict"):
            display_gif(gif_placeholder, "static/radar.gif")  
            
            time.sleep(2)  

           
            if uploaded_file is not None:
                prediction = get_prediction(model, uploaded_file)
            else:
                prediction = get_prediction(model, image_path)
            
            gif_placeholder.empty()  
            
            categories = [
                "3_long_blade_rotor", 
                "3_short_blade_rotor", 
                "Bird", 
                "Bird+mini-helicopter", 
                "drone", 
                "rc_plane"
            ]
            
            st.markdown(f"<div class='prediction-result'>The Radar has detected a: <span class='predicted-class'>{categories[prediction]}</span></div>", unsafe_allow_html=True)

        
    
    # Footer
    footer()

if __name__ == "__main__":
    main()
