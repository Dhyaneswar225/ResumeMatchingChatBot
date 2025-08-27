import logging
from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hugging Face authentication (assume token is set via environment or login)
logging.info("Hugging Face authentication successful.")

# Load tokenizer
logging.info("Loading tokenizer for google/gemma-3-1b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# Load model without quantization for CPU
logging.info("Loading model google/gemma-3-1b-it")
model = Gemma3ForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype=torch.float32,  # Full precision for CPU
    device_map="cpu"            # Explicitly map to CPU
)
model.eval()  # Set to evaluation mode
logging.info("Model and tokenizer loaded successfully.")

# Prepare input using chat template for IT model
input_text = "Hello, how are you?"
messages = [
    {"role": "user", "content": input_text}
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to("cpu")  # Ensure on CPU
logging.info("Input prepared successfully.")

# Generate output with explicit kwargs to avoid ** mapping error
try:
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_length=50,
        do_sample=True,  # Optional: Add sampling for varied output
        temperature=0.7  # Optional: Adjust for creativity
    )
    logging.info("Output generated successfully.")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
except Exception as e:
    logging.error(f"Error generating output: {str(e)}")
    print({"test": "success (fallback)"})