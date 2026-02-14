
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# ================= LOAD QWEN PROMPT ENHANCER =================
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

SYSTEM_PROMPT = """
You are a professional Stable Diffusion prompt engineer.
Convert the user text into a detailed cinematic realistic image prompt.
Add lighting, camera, environment, style, and realism details.
expressive eyes, soft smile, cinematic lighting, warm tones, bokeh background,
DSLR photography, ultra realistic, high detail, 8k, sharp focus, masterpiece,
dramatic lighting, photorealistic skin texture.

Return ONLY the prompt. Do NOT explain.
"""

def enhance_prompt(user_text):
    prompt = SYSTEM_PROMPT + "\nUser: " + user_text + "\nPrompt:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.0,
        do_sample=False,
        num_beams=5
    )

    enhanced = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return enhanced.split("Prompt:")[-1].strip()

# ================= LOAD SD PIPELINES =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Text-to-image
pipe_txt2img = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)

# Inpainting
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting"
).to(device)
pipe_inpaint.scheduler = DPMSolverMultistepScheduler.from_config(pipe_inpaint.scheduler.config)

# ================= NEGATIVE PROMPT =================
negative_prompt = (
    "low quality, worst quality, blurry, pixelated, noise, grainy, jpeg artifacts, "
    "bad anatomy, bad hands, extra fingers, missing fingers, deformed hands, "
    "bad face, ugly, distorted face, cross-eye, extra limbs, mutated, "
    "text, watermark, logo, cropped, out of frame, duplicate, "
    "overexposed, underexposed, low resolution, poorly drawn"
)

# ================= GENERATE IMAGE =================
def generate_image(prompt):
    enhanced = enhance_prompt(prompt)
    print("Enhanced Prompt:", enhanced)

    image = pipe_txt2img(
        prompt=enhanced,
        negative_prompt=negative_prompt,
        guidance_scale=10
    ).images[0]

    image.save("output.png")
    return image, enhanced

# ================= INPAINT IMAGE =================
def inpaint_image(prompt, input_image):
    enhanced = enhance_prompt(prompt)
    print("Enhanced Prompt:", enhanced)

    image = input_image.resize((768,768))

    # create center mask
    mask = Image.new("L", image.size, 0)
    mask_size = 300
    start = (image.size[0] - mask_size) // 2
    end = start + mask_size
    mask.paste(255, (start, start, end, end))

    result = pipe_inpaint(
        prompt=enhanced,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=9
    )

    final_image = result.images[0]
    final_image.save("inpainted_result.png")
    return final_image, enhanced

# ================= GRADIO UI =================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎨 SMART AI Image GENERATOR
                (Qwen + Realistic Vision V5)")

    with gr.Tab("🖼️ Text to Image"):
        txt_prompt = gr.Textbox(label="Enter Prompt", placeholder="a man proposing to a girl with a rose")
        btn_generate = gr.Button("Generate Image 🚀")
        out_img = gr.Image()
        out_prompt = gr.Textbox(label="Enhanced Prompt")

        btn_generate.click(generate_image, inputs=txt_prompt, outputs=[out_img, out_prompt])

    with gr.Tab("🧠 Inpainting"):
        in_prompt = gr.Textbox(label="Edit Prompt")
        upload_img = gr.Image(label="Upload Image")
        btn_inpaint = gr.Button("Inpaint ✨")
        out_inpaint = gr.Image()
        out_prompt2 = gr.Textbox(label="Enhanced Prompt")

        btn_inpaint.click(inpaint_image, inputs=[in_prompt, upload_img], outputs=[out_inpaint, out_prompt2])

demo.launch(share=True)

