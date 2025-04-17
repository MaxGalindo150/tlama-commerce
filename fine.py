import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig
from datasets import Dataset
import json
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd

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

# Analyze dataset
print(f"Dataset size: {len(data)} examples")

# Check for potential issues
valid_examples = []
skipped_examples = []
for i, example in enumerate(data):
    if 'input' in example and ('target' in example or 'output' in example):
        question = example['input']
        answer = example['target'] if 'target' in example else example['output']
        
        # Validate example
        if not isinstance(question, str) or not isinstance(answer, str):
            skipped_examples.append((i, "Non-string input or output"))
            continue
            
        if len(question.strip()) < 2 or len(answer.strip()) < 2:
            skipped_examples.append((i, "Too short input or output"))
            continue
        
        valid_examples.append({
            "input_text": question,
            "target_text": answer
        })
    else:
        skipped_examples.append((i, "Missing input or target/output"))

print(f"Valid examples: {len(valid_examples)}")
print(f"Skipped examples: {len(skipped_examples)}")

if skipped_examples:
    print("Sample of skipped examples:")
    for i in range(min(5, len(skipped_examples))):
        print(f"  Example {skipped_examples[i][0]}: {skipped_examples[i][1]}")

# Analyze text lengths
input_lengths = [len(ex["input_text"]) for ex in valid_examples]
target_lengths = [len(ex["target_text"]) for ex in valid_examples]

print(f"Input text length: min={min(input_lengths)}, max={max(input_lengths)}, avg={sum(input_lengths)/len(input_lengths):.1f}")
print(f"Target text length: min={min(target_lengths)}, max={max(target_lengths)}, avg={sum(target_lengths)/len(target_lengths):.1f}")

# Print some examples to verify
print("\nDataset examples:")
for i in range(min(3, len(valid_examples))):
    print(f"Example {i+1}:")
    print(f"  Input: {valid_examples[i]['input_text']}")
    print(f"  Target: {valid_examples[i]['target_text'][:100]}...")

# Shuffle and convert to Hugging Face datasets format
random.shuffle(valid_examples)
train_dataset = Dataset.from_list(valid_examples)

# Split the dataset into training and validation
train_test_split = train_dataset.train_test_split(test_size=0.1, seed=seed)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"\nTraining examples: {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}")

# Load the model and tokenizer
# For Spanish tasks, consider using mT5 which handles multiple languages
model_id = "google/mt5-large"  # Try mT5 for better multilingual support
print(f"Loading model: {model_id}")

# Check if we need model config modifications
config = AutoConfig.from_pretrained(model_id)
print(f"Model config: {config}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set the padding token if not explicitly set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Ensure special tokens are handled properly
print(f"Special tokens: {tokenizer.all_special_tokens}")
print(f"Vocab size: {tokenizer.vocab_size}")

# Load the model with fp32 for stability
print("Loading model with fp32 precision for improved stability")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    config=config,
    device_map="auto" if device == "cuda" else None
)

# Very important for T5: check that the model has the right vocab size
if model.config.vocab_size != tokenizer.vocab_size:
    print(f"Warning: Model vocab size ({model.config.vocab_size}) != Tokenizer vocab size ({tokenizer.vocab_size})")
    # Resize model embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))

# Define a correct prompt prefix for this task (Spanish generation)
prompt_prefix = "pregunta: "
answer_prefix = "respuesta: "

# Improved tokenize function with proper prefix for Spanish tasks
def tokenize_function(examples):
    # Prepare inputs with correct task prefix for Spanish
    inputs = [prompt_prefix + text for text in examples["input_text"]]
    targets = [answer_prefix + text for text in examples["target_text"]]
    
    # Tokenize inputs with fixed max length
    input_encoding = tokenizer(
        inputs, 
        padding="max_length",
        max_length=128,  # Adjust based on your input length analysis
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize targets separately
    target_encoding = tokenizer(
        targets,
        padding="max_length",
        max_length=512,  # Adjust based on target length analysis
        truncation=True,
        return_tensors="pt"
    )
    
    # Replace pad token with -100 for loss calculation
    labels = target_encoding["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Combine into a single batch
    model_inputs = {
        "input_ids": input_encoding["input_ids"],
        "attention_mask": input_encoding["attention_mask"],
        "labels": labels
    }
    
    return model_inputs

print("\nTokenizing datasets...")
tokenized_train = train_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=train_dataset.column_names,
    batch_size=16
)
tokenized_eval = eval_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=eval_dataset.column_names,
    batch_size=16
)

# Verify tokenized data
print("\nVerifying tokenized data:")
sample = tokenized_train[0]
print("Input IDs sample:", sample["input_ids"][:10])
print("Input text decoded:", tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
print("Label sample:", [l for l in sample["labels"][:10] if l != -100])
print("Label text decoded:", tokenizer.decode([l for l in sample["labels"] if l != -100], skip_special_tokens=True))

# Training args with more conservative settings
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-spanish-generation",
    per_device_train_batch_size=2,  # Smaller batch size for stability
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Maintain effective batch size of 16
    learning_rate=2e-5,  # Very conservative learning rate
    num_train_epochs=5,
    fp16=False,  # Use fp32 for stability
    logging_dir="./logs-mt5",
    logging_steps=10,
    save_steps=50,
    eval_steps=50,
    save_total_limit=3,
    predict_with_generate=True,
    generation_max_length=512,
    generation_num_beams=4,
    gradient_checkpointing=True,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    eval_strategy="steps",
    max_grad_norm=1.0,  # Gradient clipping
    seed=seed,
)

# Define better metrics calculation
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Remove prefixes for cleaner comparison
    cleaned_preds = [pred.replace(answer_prefix, "").strip() for pred in decoded_preds]
    cleaned_labels = [label.replace(answer_prefix, "").strip() for label in decoded_labels]
    
    # Calculate metrics
    exact_matches = sum(pred == label for pred, label in zip(cleaned_preds, cleaned_labels))
    exact_match_accuracy = exact_matches / len(cleaned_preds)
    
    # Calculate BLEU or other metrics here if needed
    
    # Print a few examples for visual inspection
    print("\nPrediction samples:")
    for i in range(min(3, len(cleaned_preds))):
        input_text = eval_dataset[i]["input_text"]
        print(f"Input: {input_text}")
        print(f"Pred: {cleaned_preds[i][:100]}...")
        print(f"True: {cleaned_labels[i][:100]}...")
        print()
    
    return {
        "exact_match_accuracy": exact_match_accuracy
    }

# Create tensorboard writer
tb_writer = SummaryWriter(log_dir=training_args.logging_dir)

# Early stopping callback to prevent overfitting
from transformers.trainer_callback import EarlyStoppingCallback
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

# Configure the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

# Enable debug mode for more informative errors
torch.autograd.set_detect_anomaly(True)

# Function to test model generation
def test_model_generation(model, tokenizer, input_texts, device):
    model.eval()
    results = []
    
    for input_text in input_texts:
        # Prepare input
        encoded_input = tokenizer(
            prompt_prefix + input_text, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids=encoded_input.input_ids,
                attention_mask=encoded_input.attention_mask,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        # Clean prefix if present
        if decoded_output.startswith(answer_prefix):
            decoded_output = decoded_output[len(answer_prefix):].strip()
            
        results.append({
            "input": input_text,
            "output": decoded_output
        })
    
    return results

# Start training with proper error handling
try:
    print("\nStarting training...")
    trainer.train()
    
    # Save the model
    model_path = "./mt5-spanish-generation-final"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")
    
    # Test the model on examples
    print("\nTesting trained model on examples:")
    test_inputs = [ex["input_text"] for ex in eval_dataset.select(range(5))]
    test_outputs = test_model_generation(model, tokenizer, test_inputs, device)
    
    for i, result in enumerate(test_outputs):
        print(f"Input: {result['input']}")
        print(f"Generated: {result['output']}")
        print(f"Expected: {eval_dataset[i]['target_text'][:100]}...")
        print()

except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()