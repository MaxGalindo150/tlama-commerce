import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.integrations import TensorBoardCallback
from datasets import Dataset
import json
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import random

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Configure device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the dataset
with open('eigencore_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare the dataset for T5
train_data = []
for example in data:
    if 'input' in example and ('target' in example or 'output' in example):
        question = example['input']
        answer = example['target'] if 'target' in example else example['output']
        train_data.append({
            "input_text": question,
            "target_text": answer
        })

# Shuffle the data
random.shuffle(train_data)

# Print some examples to verify
print("Dataset examples:")
for i in range(min(3, len(train_data))):
    print(f"Example {i+1}:")
    print(f"  Input: {train_data[i]['input_text'][:100]}...")
    print(f"  Target: {train_data[i]['target_text'][:100]}...")

# Convert to Hugging Face datasets format
train_dataset = Dataset.from_list(train_data)

# Split the dataset into training and validation
train_test_split = train_dataset.train_test_split(test_size=0.1, seed=seed)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}")

# Load the model and tokenizer
model_id = "google/flan-t5-large"  # Consider starting with base model if memory issues
print(f"Loading model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model with fp32 for stability in initial training
print("Loading model with fp32 precision for improved stability")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    device_map="auto" if device == "cuda" else None
)

# Create a callback for gradient monitoring
class GradientMonitorCallback(TensorBoardCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 100 == 0:  # Log every 100 steps
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer = self.tb_writer
                    writer.add_histogram(f"gradients/{name}", param.grad, state.global_step)
                    writer.add_scalar(f"gradient_norm/{name}", param.grad.norm().item(), state.global_step)

# Tokenize the dataset with improved handling
def tokenize_function(examples):
    # Prepare input with task prefix
    inputs = ["translate Spanish to English: " + text for text in examples["input_text"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        padding="max_length", 
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize targets - don't pad to max_length
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"], 
            max_length=512, 
            truncation=True,
            padding=False
        )
    
    # Convert to tensors and assign to inputs
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# Verify tokenized data
print("Verifying tokenized data:")
sample = tokenized_train[0]
print("Input IDs sample:", sample["input_ids"][:10])
print("Input text decoded:", tokenizer.decode(sample["input_ids"], skip_special_tokens=True)[:50])
print("Label IDs sample:", sample["labels"][:10])
print("Label text decoded:", tokenizer.decode([l for l in sample["labels"] if l != -100], skip_special_tokens=True)[:50])

# Function to prepare batched labels properly
def data_collator(features):
    batch = {}
    
    # Prepare input_ids, attention_mask as usual
    input_keys = ["input_ids", "attention_mask"]
    for key in input_keys:
        if key not in features[0]:
            continue
        batch[key] = torch.stack([torch.tensor(f[key]) for f in features])
    
    # Handle labels specially - pad to max length in batch
    if "labels" in features[0]:
        label_lengths = [len(f["labels"]) for f in features]
        max_label_length = max(label_lengths)
        
        labels = []
        for f in features:
            padding_length = max_label_length - len(f["labels"])
            labels.append(
                f["labels"] + [-100] * padding_length
            )
        
        batch["labels"] = torch.tensor(labels)
    
    return batch

# Configure training arguments with lower learning rate and gradient clipping
training_args = Seq2SeqTrainingArguments(
    output_dir="./flan-t5-large-eigencore-fixed",
    per_device_train_batch_size=4,  # Smaller batch size for stability
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Increased to maintain effective batch size
    learning_rate=5e-5,  # Lower learning rate
    num_train_epochs=5,
    fp16=False,  # Start with fp32 for stability
    bf16=False,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    save_total_limit=3,
    predict_with_generate=True,
    gradient_checkpointing=True,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",  # Changed to linear for more stable warmup
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    eval_strategy="steps",
    max_grad_norm=1.0,  # Add gradient clipping
    generation_max_length=512,
    generation_num_beams=4,
    seed=seed,
)

# Define metrics calculation
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate exact match accuracy
    exact_matches = sum(pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels))
    exact_match_accuracy = exact_matches / len(decoded_preds)
    
    # Print a few examples for visual inspection
    print("\nPrediction samples:")
    for i in range(min(3, len(decoded_preds))):
        print(f"Input: {tokenizer.decode(eval_dataset[i]['input_text'], skip_special_tokens=True)}")
        print(f"Pred: {decoded_preds[i]}")
        print(f"True: {decoded_labels[i]}")
        print()
    
    return {
        "exact_match_accuracy": exact_match_accuracy
    }

# Create tensorboard writer
tb_writer = SummaryWriter(log_dir=training_args.logging_dir)

# Configure the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[GradientMonitorCallback(tb_writer=tb_writer)],
)

# Enable model debugging
torch.autograd.set_detect_anomaly(True)

# Start training with proper error handling
try:
    print("Starting training...")
    trainer.train()
    
    # Save the model
    model_path = "./flan-t5-large-eigencore-fixed-final"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")
    
    # Test the model on a few examples
    print("\nTesting trained model on a few examples:")
    for i in range(min(5, len(eval_dataset))):
        input_text = eval_dataset[i]["input_text"]
        inputs = tokenizer("translate Spanish to English: " + input_text, return_tensors="pt").to(device)
        
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {input_text}")
        print(f"Prediction: {prediction}")
        print(f"Target: {eval_dataset[i]['target_text']}")
        print()

except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()