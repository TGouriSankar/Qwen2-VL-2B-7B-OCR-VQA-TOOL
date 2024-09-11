import requests

API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2-VL-7B-Instruct"
headers = {"Authorization": "Bearer <hugging-face token>"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("/media/player/karna1/qwen2-VL-7B/cg.jpg")
print(output)


# <<<<<<<<<<<------------------------Run it in local server------------------------>>>>>>>>>>>>

# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load the model and tokenizer
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# # Generate text (example for text input)
# inputs = tokenizer("Your input here", return_tensors="pt")
# outputs = model.generate(**inputs)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
