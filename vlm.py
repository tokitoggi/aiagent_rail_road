from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP model (VLM)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")

# Generate description or answer a question based on image

def analyze_with_vlm(image: Image.Image, question: str = None) -> str:
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    if question:
        prompt = f"Question: {question} Answer:"
    else:
        prompt = "Describe the scene."

    outputs = model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption
