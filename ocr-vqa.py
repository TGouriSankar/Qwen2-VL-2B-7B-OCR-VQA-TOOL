from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from PIL import Image
import requests

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float32,
    device_map="auto",  # Try to map it optimally for your system
)

print("Model loaded.")
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct"
)
image_path = "/media/player/karna1/qwen2-VL-7B/image.png"
image = Image.open(image_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": "extract the text from the image"
            }
        ]
    }
]

text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

print("Processing image and text prompt...")
inputs = processor(
    text = [text_prompt],
    images = [image],
    padding = True,
    return_tensors = "pt"
)

print("Processing complete.")
inputs = inputs.to("cpu")

print("Running inference...")
output_ids = model.generate(**inputs, max_new_tokens=64) #1024 or 128 or 512
print("Inference complete.")

generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]

output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)