import time
import json
import pandas as pd
import numpy as np
import torch
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

def get_gpu_utilization():
    """Check GPU utilization using nvidia-smi"""
    import subprocess
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        return float(output.decode().strip()) / 100 
    except Exception as e:
        print(f"Error retrieving GPU utilization: {e}")
        return np.nan

def generate_labels(task_prompt, image, text_input=None, model=None, processor=None, device=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    output = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return output

def process_image(image_path, model, processor, device, task_prompts, text_inputs=None):
    """Process a single image (URL or local path) and return results."""
    start_time = time.time()

    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(image_path).convert('RGB')

    results = {}
    for i, (task_prompt, text_input) in enumerate(zip(task_prompts, text_inputs or ['']*len(task_prompts))):
        result = generate_labels(task_prompt, img, text_input=text_input, model=model, processor=processor, device=device)
        results[f'task_{i+1}_result'] = result

    elapsed_time = time.time() - start_time
    gpu_util = get_gpu_utilization()

    return results, elapsed_time, gpu_util

def save_results_to_json(data, file_path):
    """Save results to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

json_file = '/content/drive/MyDrive/CDAC_image_capuring/image_processing_payload.json' 
with open(json_file, 'r') as file:
    json_data = json.load(file)

image_paths = [item['image_link'] for item in json_data]
post_ids = [item['post_id'] for item in json_data]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'microsoft/Florence-2-base'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(model_id, device_map=device, trust_remote_code=True)

task_prompts = ['<CAPTION>', '<MORE_DETAILED_CAPTION>', '<OD>', '<CAPTION_TO_PHRASE_GROUNDING>']
text_inputs = [None, None, None, 'man'] 
results = []

for image_path, post_id in zip(image_paths, post_ids):
    task_results, elapsed_time, gpu_util = process_image(image_path, model, processor, device, task_prompts, text_inputs)
    image_description = task_results.get('task_2_result', {}).get('<MORE_DETAILED_CAPTION>', 'No Detailed Caption')
    result_row = {
        'post_id': post_id,
        'image_link': image_path,
        'image_description': image_description
    }
    results.append(result_row)

    print(f"Post ID: {post_id}")
    print(f"Image Link: {image_path}")
    print(f"Image Description: {image_description}")
    print("-" * 40)

save_results_to_json(results, '/content/drive/MyDrive/CDAC_image_capuring/output_results.json')
