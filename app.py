import os
from dotenv import load_dotenv
import streamlit as st
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import PIL
from camera_input_live import camera_input_live

st.set_page_config(layout="wide")

class ImageConverter:
    def __init__(self, model_id="timbrooks/instruct-pix2pix"):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
        self.pipe.to("cpu")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
    def convert(self, prompt, original_image):
        if not original_image:
            st.error("Click 'Take Photo' and wait for it to load before trying the image generation options")
            return
        image = PIL.Image.open(original_image)
        image = image.resize((352, 626))
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        
        converted_images = self.pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
        return converted_images[0]

class App:
    def __init__(self):
        load_dotenv()
        self.image_converter = ImageConverter()
        
        if 'original_image' not in st.session_state:
            st.session_state.original_image = None
    
    def run(self):
        self._setup_layout()

    def _setup_layout(self):
        st.markdown("""
        <style>
            .css-19rxjzo {
                width: 100%;
            }
            .css-1r6slb0{
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.header("Original")
        col2.header("Options")
        col3.header("Processed")
        options = ["Anime", "Cartoon", "Sketch", "Line Art", "Cyborg"]
        
        for option in options:
            if col2.button(option):
                converted_image = self.image_converter.convert(f"Convert the image to a {option}", st.session_state.original_image)
                if converted_image:
                    col3.image(converted_image)
        st.session_state.original_image = col1.camera_input("Take a photo",label_visibility="hidden")
        
if __name__ == "__main__":
    app = App()
    app.run()
