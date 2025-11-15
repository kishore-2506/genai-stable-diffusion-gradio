## Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

### PROBLEM STATEMENT:
Design a deep learning–based text-to-image generation system using the Stable Diffusion model, providing a simple and interactive interface for users to enter prompts and generate images dynamically.
### DESIGN STEPS:

Step 1: Load the Hugging Face API key and required Python libraries (Gradio, dotenv, PIL, requests).
Step 2: Configure the API endpoint and define helper functions for making POST requests to the Stable Diffusion API and decoding the base64 image output.
Step 3: Create a Gradio interface with text input for the prompt, sliders for model parameters (steps, guidance, width, height), and image output for visualization.
Step 4: Launch the Gradio app in a Jupyter environment with shareable link for real-time testing and evaluation.

### PROGRAM:
```
import os
import io
import IPython.display
from PIL import Image
import base64
from dotenv import load_dotenv, find_dotenv
import requests, json
import gradio as gr

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']


def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_TTI_BASE']):
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST", ENDPOINT_URL, headers=headers, data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))


def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image


def generate(prompt, negative_prompt, steps, guidance, width, height):
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    output = get_completion(prompt, params)
    result_image = base64_to_pil(output)
    return result_image


gr.close_all()
with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with Stable Diffusion")
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Your prompt")
        with gr.Column(scale=1, min_width=50):
            btn = gr.Button("Submit")
    with gr.Accordion("Advanced options", open=False):
        negative_prompt = gr.Textbox(label="Negative prompt")
        with gr.Row():
            with gr.Column():
                steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25)
                guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7)
            with gr.Column():
                width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
                height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
    output = gr.Image(label="Result")
    btn.click(fn=generate, inputs=[prompt, negative_prompt, steps, guidance, width, height], outputs=[output])

gr.close_all()
demo.launch(share=True, server_port=int(os.environ.get("PORT", 7860)))
gracefully_close = True
```
### OUTPUT:

<img width="915" height="496" alt="exp 7 1" src="https://github.com/user-attachments/assets/ef2ca420-18b4-4048-94c2-99354566d3ce" />

<img width="917" height="498" alt="exp 7 2" src="https://github.com/user-attachments/assets/5cdd9f74-6d67-4644-9349-5bbf2460833b" />

### RESULT:
An interactive image generation prototype using the Stable Diffusion model and Gradio framework was successfully developed and executed.
