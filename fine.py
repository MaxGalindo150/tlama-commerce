import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
import json
import numpy as np
import os

# Configurar para usar dispositivo adecuado
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Cargar el dataset
with open('eigencore_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Preparar el dataset para T5
train_data = []
for example in data:
    if 'input' in example and ('target' in example or 'output' in example):
        question = example['input']
        answer = example['target'] if 'target' in example else example['output']
        train_data.append({
            "input_text": question,
            "target_text": answer
        })

# Convertir a formato de Hugging Face datasets
train_dataset = Dataset.from_list(train_data)

# Dividir el dataset en entrenamiento y validación
train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Cargar el modelo y tokenizer
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Cargar el modelo con fp16 para eficiencia
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto" if device == "cuda" else None
)

# Tokenizar el dataset
def tokenize_function(examples):
    # T5 espera que la entrada tenga un prefijo que indique la tarea
    inputs = ["responde: " + text for text in examples["input_text"]]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        padding="max_length", 
        truncation=True
    )
    
    labels = tokenizer(
        examples["target_text"], 
        max_length=512, 
        padding="max_length", 
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    
    # Reemplazar padding token id con -100 en labels para que no se consideren en la pérdida
    for i in range(len(model_inputs["labels"])):
        model_inputs["labels"][i] = [
            -100 if token == tokenizer.pad_token_id else token 
            for token in model_inputs["labels"][i]
        ]
    
    return model_inputs

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Verificar la estructura de los datos
sample = tokenized_train[0]
print("Input IDs:", sample["input_ids"][:20])
print("Labels:", [l for l in sample["labels"][:20] if l != -100])

# Configurar los argumentos de entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir="./flan-t5-large-eigencore-full",
    per_device_train_batch_size=8,  # Ajustar según memoria disponible
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Acumular gradientes para compensar batch pequeño
    learning_rate=1e-4,
    num_train_epochs=3,
    fp16=True if device == "cuda" else False,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    save_total_limit=3,
    predict_with_generate=True,
    gradient_checkpointing=True,  # Activar para ahorrar memoria
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",  # Desactivar reporting a wandb u otros
    eval_strategy="steps",
)

# Definir una función de cálculo de métricas
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    # Decodificar predicciones
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Reemplazar -100 en labels por el pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calcular exactitud exacta (exact match)
    exact_matches = sum(pred == label for pred, label in zip(decoded_preds, decoded_labels))
    exact_match_accuracy = exact_matches / len(decoded_preds)
    
    return {"exact_match_accuracy": exact_match_accuracy}

# Configurar el entrenador
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Iniciar entrenamiento
trainer.train()

# Guardar el modelo entrenado
model_path = "./flan-t5-large-eigencore-full-final"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print(f"Modelo guardado en {model_path}")