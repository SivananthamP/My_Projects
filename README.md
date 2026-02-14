🎨 Smart AI Image Generator
Qwen + Stable Diffusion (Realistic Vision V5) + Inpainting + Gradio UI
Smart AI Image Generator is an advanced AI-powered image generation application that enhances user prompts using a large language model and generates ultra-realistic images using Stable Diffusion.
This project combines:
🤖 Qwen 2.5-1.5B-Instruct – For automatic cinematic prompt enhancement
🖼️ Realistic Vision V5.1 (Stable Diffusion) – For high-quality text-to-image generation
🧠 Stable Diffusion Inpainting – For intelligent image editing
🌐 Gradio UI – For a clean and interactive web interface
🚀 Features
✨ 1. AI Prompt Enhancement
User input is automatically transformed into a detailed, cinematic, ultra-realistic prompt including:
Lighting and environment details
Camera settings
DSLR photography style
Skin texture and realism
8K high detail rendering
Powered by Qwen LLM, ensuring professional-level prompt engineering.
🖼️ 2. Text-to-Image Generation
Uses Realistic Vision V5.1
Applies optimized negative prompts to remove:
Blurry outputs
Bad anatomy
Extra fingers
Distortions
Watermarks
High guidance scale for improved accuracy
Automatically saves generated images
🧠 3. AI Inpainting
Upload an image
Provide an edit prompt
The model intelligently regenerates the masked center area
Maintains realistic blending and lighting consistency
🛠️ Tech Stack
Python
PyTorch
Diffusers
Transformers
Gradio
Stable Diffusion
📦 Installation
Run once:
Copy code
Bash
pip install diffusers transformers accelerate safetensors gradio
▶️ How It Works
User enters a simple prompt
Qwen enhances it into a cinematic AI-ready prompt
Stable Diffusion generates a high-quality image
(Optional) Inpainting modifies selected image regions
🖥️ User Interface
The app provides two tabs:
🖼️ Text-to-Image
🧠 Inpainting
Launches with a shareable Gradio link.
🎯 Use Cases
AI Art Creation
Content Creation
Thumbnail Design
Social Media Graphics
Concept Art
Creative Prototyping
💡 Why This Project?
Most users struggle with writing good Stable Diffusion prompts.
This project automatically converts simple ideas into professional cinematic prompts, making high-quality AI art accessible to everyone.
If you want, I can also give:
🔥 A shorter README version
📱 A mobile-friendly README
⭐ A more professional portfolio-style description
🧠 A version optimized for recruiters