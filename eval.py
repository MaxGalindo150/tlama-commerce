import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import time
from rouge_score import rouge_scorer
import os
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("--model_path", type=str, default="./flan-t5-large-eigencore-full-final",
                        help="Path to the saved fine-tuned model")
    parser.add_argument("--base_model_path", type=str, default="google/flan-t5-large",
                        help="Path to the base model for comparison")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Path to a JSON file with test questions (optional)")
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32", "bf16"], default="fp16",
                        help="Precision for model inference")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum length for generation")
    parser.add_argument("--prefix", type=str, default="responde: ",
                        help="Prefix to add to input questions for fine-tuned model")
    return parser.parse_args()

def load_models(args):
    # Configure dtype based on precision argument
    if args.precision == "fp16" and torch.cuda.is_available():
        dtype = torch.float16
    elif args.precision == "bf16" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, precision: {args.precision}")
    
    # Load the fine-tuned model
    print(f"Loading fine-tuned model from {args.model_path}")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None
        )
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        print("Falling back to CPU with fp32")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32
        )
    
    # Load the base model for comparison
    print(f"Loading base model from {args.base_model_path}")
    try:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("Falling back to CPU with fp32")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float32
        )
    
    # Load the tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    return base_model, model, tokenizer

def load_test_questions(test_file):
    import json
    
    if test_file and os.path.exists(test_file):
        print(f"Loading test questions from {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        references = []
        
        for item in data:
            if isinstance(item, dict) and 'input' in item:
                questions.append(item['input'])
                if 'target' in item:
                    references.append(item['target'])
                elif 'output' in item:
                    references.append(item['output'])
                else:
                    references.append("")  # No reference available
        
        return questions, references
    
    # Default questions if no test file is provided
    return [
        "¿Qué es EigenCore?",
        "¿Dónde está ubicada la empresa?",
        "¿Cuáles son los servicios que ofrecen?",
        "¿Qué es Tlama?",
        "¿Cómo protegen los datos de sus clientes?",
        "k es eigencore?",  # Pregunta con errores ortográficos para probar robustez
    ], []

def generate_response(model, tokenizer, input_text, prefix="", max_length=512, device=None):
    # Prepare the input
    input_with_prefix = prefix + input_text
    
    # Tokenize
    inputs = tokenizer(input_with_prefix, return_tensors="pt")
    input_ids = inputs.input_ids
    
    if device:
        input_ids = input_ids.to(device)
    
    # Generate response with error handling
    try:
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        generation_time = time.time() - start_time
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response, generation_time
    except Exception as e:
        print(f"Error during generation: {e}")
        return f"Error: {str(e)}", 0

def calculate_metrics(predictions, references):
    if not references or len(references) == 0:
        return None
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        if not ref:  # Skip if no reference is available
            continue
        
        score = scorer.score(pred, ref)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    
    # Calculate average scores
    avg_scores = {key: sum(values)/len(values) if values else 0 for key, values in scores.items()}
    return avg_scores

def main():
    args = parse_arguments()
    base_model, model, tokenizer = load_models(args)
    questions, references = load_test_questions(args.test_file)
    
    device = next(model.parameters()).device
    
    print("EVALUACIÓN DEL MODELO FINE-TUNED:")
    print("-" * 50)
    
    fine_tuned_responses = []
    base_responses = []
    total_ft_time = 0
    total_base_time = 0
    
    for idx, question in enumerate(tqdm(questions, desc="Evaluating")):
        print(f"\nPREGUNTA {idx+1}: {question}")
        
        # Generate response with the fine-tuned model
        response, ft_time = generate_response(
            model, tokenizer, question, prefix=args.prefix, 
            max_length=args.max_length, device=device
        )
        fine_tuned_responses.append(response)
        total_ft_time += ft_time
        print(f"RESPUESTA (Fine-tuned - {ft_time:.2f}s): {response}")
        
        # Generate response with the base model for comparison
        print("-" * 30)
        base_response, base_time = generate_response(
            base_model, tokenizer, question, prefix="", 
            max_length=args.max_length, device=device
        )
        base_responses.append(base_response)
        total_base_time += base_time
        print(f"RESPUESTA DEL MODELO BASE ({base_time:.2f}s): {base_response}")
        print("-" * 50)
    
    # Print timing statistics
    print("\nESTADÍSTICAS DE GENERACIÓN:")
    print(f"Tiempo total modelo fine-tuned: {total_ft_time:.2f}s")
    print(f"Tiempo promedio por pregunta (fine-tuned): {total_ft_time/len(questions):.2f}s")
    print(f"Tiempo total modelo base: {total_base_time:.2f}s")
    print(f"Tiempo promedio por pregunta (base): {total_base_time/len(questions):.2f}s")
    
    # Calculate and print ROUGE scores if references are available
    if references and len(references) > 0:
        ft_metrics = calculate_metrics(fine_tuned_responses, references)
        base_metrics = calculate_metrics(base_responses, references)
        
        print("\nMÉTRICAS DE EVALUACIÓN:")
        print("Fine-tuned model:")
        for key, value in ft_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("Base model:")
        for key, value in base_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # Additional out-of-domain test
    print("\nPRUEBA FUERA DE DOMINIO:")
    ood_question = "¿Cuál es la capital de Francia?"
    print(f"PREGUNTA: {ood_question}")
    
    ood_response, ood_time = generate_response(
        model, tokenizer, ood_question, prefix=args.prefix,
        max_length=args.max_length, device=device
    )
    print(f"RESPUESTA (Fine-tuned - {ood_time:.2f}s): {ood_response}")

if __name__ == "__main__":
    main()