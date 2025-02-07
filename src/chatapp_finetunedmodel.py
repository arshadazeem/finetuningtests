from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model
model_path = r"C:\Users\arazeem\source\finetuning-tests\fine_tuned_tinyllama"  

# Update this with the actual path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move to GPU if available
device = "cpu"
model.to(device)

def chat():
    print("🤖 TinyLlama Chat: Type 'exit' to stop.")
    
    while True:
        user_input = input("\nYou: ")  # Get user input
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        prompt = f"You are friendly chatbot that responds to this user input or user questions: {user_input}\nAI:"    

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            output = model.generate(**inputs, max_length=200, temperature=0.7,   # Adds randomness
            top_p=0.9,         # Nucleus sampling
            top_k=50,          # Consider only top-k highest probability words
            do_sample=True,    # Enable sampling
            pad_token_id=tokenizer.eos_token_id)

        # Decode and print response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"🤖 Bot: {response}")

if __name__ == "__main__":
    chat()
