import json

# Ruta al archivo JSON
file_path = "eigencore_dataset.json"  # Cambia al nombre de tu archivo

# Leer el archivo JSON
with open(file_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# Ahora 'dataset' es una lista de diccionarios, cada uno con 'input' y 'target'
print(f"Número total de ejemplos: {len(dataset)}")

# Acceder al primer ejemplo
print(f"Ejemplo 1 - Pregunta: {dataset[0]['input']}")
print(f"Ejemplo 1 - Respuesta: {dataset[0]['target']}")