import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load the model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Define the generation function
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Launch the Gradio interface
gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="EEcho-image45",
    description="A smart image generator built by Austin using Stable Diffusion"
).launch()
