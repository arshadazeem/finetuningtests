from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from transformers import Trainer, TrainingArguments

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
#model_name = "microsoft/phi-1_5", "TinyLlama/TinyLlama_v1.1"

model = AutoModelForCausalLM.from_pretrained(model_name)  
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Assign a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

# Prepare the dataset (using Hugging Face Datasets)
file_path=r"C:\Users\arazeem\source\finetuning-tests\src\mydataset.txt"
dataset = load_dataset("text", data_files={"train": file_path})

#print(dataset["train"][0])

# Tokenizing function
def tokenize_function(example):
    inputs = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()  # Set labels equal to input_ids
    return inputs

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Print a tokenized sample
print(tokenized_dataset["train"][0])

#torch.save(tokenized_dataset, r"C:\Users\arazeem\source\finetuning-tests\my_tokenized_dataset.pt")

# setup training arguments

def train():

    training_args = TrainingArguments(
        output_dir="./saved_checkpoints",
        num_train_epochs=2,           # Number of epochs
        per_device_train_batch_size=1, # Batch size per device
        gradient_accumulation_steps=1, # For larger batch sizes
        evaluation_strategy="epoch",   # Evaluate every epoch
        save_steps=1000,                # Save checkpoint every 500 steps
        logging_dir="./logs",          # Directory for logs
        logging_steps=100,             # Log every 100 steps
        #load_best_model_at_end=True,   # Load the best model after training
        save_total_limit=1,            # Save only the latest 2 checkpoints
        fp16=False,                 # Avoid mixed precision (requires higher GPU memory)
        max_steps=-1,                      
        dataloader_num_workers=1,          # Reduce number of workers to lower CPU load
        per_device_eval_batch_size=2,      # Same batch size for evaluation
        report_to="none",                  # Avoid using any tracking (if you are not using WandB)
    )


    # Fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],  # Use the tokenized training dataset
        eval_dataset=tokenized_dataset["train"],   # You can set up an eval dataset too
        tokenizer=tokenizer,                       # Use the tokenizer
    )

    # Start the training process
    trainer.train()

if __name__ == "__main__":
    train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_tinyllama")
tokenizer.save_pretrained("./fine_tuned_tinyllama")