import torch
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM

# Ruta donde se guardó el modelo con LoRA
lora_model_path = "./flan-t5-large-eigencore-lora"

# Cargar la configuración de LoRA
config = PeftConfig.from_pretrained(lora_model_path)

# Cargar el modelo base
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Cargar el modelo con adaptadores LoRA aplicados
model = PeftModel.from_pretrained(
    base_model, 
    lora_model_path
)

# Cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Función para generar respuestas
def generate_response(model, input_text, prefix="responde: "):
    # Preparar la entrada
    input_with_prefix = prefix + input_text
    
    # Tokenizar
    input_ids = tokenizer(input_with_prefix, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    
    # Generar respuesta
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
    
    # Decodificar la respuesta
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Lista de preguntas de prueba
test_questions = [
    "¿Qué es EigenCore?",
    "¿Dónde está ubicada la empresa?",
    "¿Cuáles son los servicios que ofrecen?",
    "¿Qué es Tlama?",
    "¿Cómo protegen los datos de sus clientes?",
    "k es eigencore?",  # Pregunta con errores ortográficos para probar robustez
]

# Probar el modelo con las preguntas
print("EVALUACIÓN DEL MODELO FINE-TUNED:")
print("-" * 50)

for question in test_questions:
    print(f"\nPREGUNTA: {question}")
    
    # Generar respuesta con el modelo fine-tuned
    response = generate_response(model, question)
    print(f"RESPUESTA: {response}")
    
    # También podemos probar el modelo base para comparar
    print("-" * 30)
    base_response = generate_response(base_model, question, prefix="")
    print(f"RESPUESTA DEL MODELO BASE: {base_response}")
    print("-" * 50)

# Prueba adicional: Generar respuesta para una pregunta fuera del dominio
print("\nPRUEBA FUERA DE DOMINIO:")
out_of_domain = "¿Cuál es la capital de Francia?"
print(f"PREGUNTA: {out_of_domain}")
ood_response = generate_response(model, out_of_domain)
print(f"RESPUESTA: {ood_response}")