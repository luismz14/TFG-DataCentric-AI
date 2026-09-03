import json
from collections import Counter

def analizar_dataset_completo(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ext_counter = Counter()
        size_by_ext = Counter()
        total_files = 0

        for patient in data:
            for file in patient['files']:
                ext = file.get('extension', '.unknown').lower()
                if ext == "": ext = "no_ext"
                
                ext_counter[ext] += 1
                size_by_ext[ext] += file.get('size_mb', 0)
                total_files += 1

        print(f"{'EXTENSION':<15} | {'CANTIDAD':<10} | {'TAMAÑO TOTAL (MB)':<15}")
        print("-" * 45)
        
        for ext, count in ext_counter.most_common():
            size = size_by_ext[ext]
            print(f"{ext:<15} | {count:<10} | {size:<15.2f}")

        print("-" * 45)
        print(f"Total de archivos analizados: {total_files}")
        print(f"Total de carpetas (pacientes): {len(data)}")

    except FileNotFoundError:
        print(f"❌ Error: No se encuentra {json_file}. Ejecuta el indexador primero.")

if __name__ == "__main__":
    analizar_dataset_completo('dataset_inventory.json')