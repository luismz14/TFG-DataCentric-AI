import os
import pandas as pd
from datetime import datetime

def merge_csv_datasets(input_folder="data", output_folder="results"):
    """
    Recorre las carpetas, verifica el esquema de los CSVs y genera un reporte en TXT.
    No intenta corregir separadores automáticamente.
    
    Args:
        input_folder (str): Ruta de origen.
        output_folder (str): Ruta de destino para el CSV unido y el reporte.
    """
    
    # 1. Preparar carpeta de salida
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Lista para guardar las líneas del reporte
    report_lines = []
    
    def log(message):
        """Imprime en consola y guarda en el buffer del reporte"""
        print(message)
        report_lines.append(message)

    # Cabecera del reporte
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log("==================================================")
    log(f"📊  MERGE REPORT - {timestamp}")
    log("==================================================")
    log(f"📂 Input: '{input_folder}'")
    log(f"📂 Output: '{output_folder}'\n")

    all_dataframes = []
    mismatched_files = []
    read_errors = []
    
    # Schema de referencia (se llena con el primer archivo válido)
    standard_columns = None
    reference_file = None
    
    # 2. Recorrido recursivo
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                # El ID es el nombre de la carpeta contenedora
                patient_id = os.path.basename(root)

                try:
                    # Lectura estándar (sin lógica extra de separadores)
                    df = pd.read_csv(file_path)
                    
                    # Validar archivo vacío
                    if df.empty:
                        log(f"⚠️  Skipping empty file: {patient_id}/{file}")
                        continue

                    # 3. Establecer Referencia (Primer archivo válido dicta la norma)
                    if standard_columns is None:
                        standard_columns = list(df.columns)
                        reference_file = f"{patient_id}/{file}"
                        log(f"✅ Reference Schema Established from: {reference_file}")
                        log(f"   Columns: {standard_columns}\n")
                    
                    # 4. Comparación Estricta
                    current_columns = list(df.columns)
                    
                    if current_columns == standard_columns:
                        # Añadimos trazabilidad
                        df['source_patient_id'] = patient_id
                        df['source_filename'] = file
                        all_dataframes.append(df)
                    else:
                        # Registramos el fallo para el reporte
                        missing = set(standard_columns) - set(current_columns)
                        extra = set(current_columns) - set(standard_columns)
                        mismatched_files.append({
                            "file": f"{patient_id}/{file}",
                            "missing": list(missing),
                            "extra": list(extra)
                        })
                        
                except Exception as e:
                    read_errors.append(f"{patient_id}/{file}: {str(e)}")

    # --- GENERACIÓN DEL REPORTE TXT ---
    
    log("\n" + "="*50)
    log("📝  DETAILED ANALYSIS")
    log("="*50)
    
    if mismatched_files:
        log(f"❌ FOUND {len(mismatched_files)} FILES WITH SCHEMA MISMATCH:")
        for error in mismatched_files:
            log(f"  - File: {error['file']}")
            if error['missing']: 
                log(f"    └── Missing Cols: {error['missing']}")
            if error['extra']:   
                log(f"    └── Extra Cols:   {error['extra']}")
            log("") # Línea en blanco para separar
    else:
        log("✅ All processed files match the reference schema!")

    if read_errors:
        log(f"\n🚫 CORRUPT FILES ({len(read_errors)}):")
        for err in read_errors:
            log(f"  - {err}")

    # Guardar el reporte físico
    report_path = os.path.join(output_folder, "merge_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")
            
    print(f"\n📄 Report text file saved to: {report_path}")

    # --- MERGE FINAL ---
    if all_dataframes:
        log(f"\n🔄 Merging {len(all_dataframes)} valid files...")
        master_df = pd.concat(all_dataframes, ignore_index=True)
        
        csv_path = os.path.join(output_folder, "master_dataset.csv")
        master_df.to_csv(csv_path, index=False)
        
        print(f"🎉 SUCCESS! Merged dataset saved to: {csv_path}")
        print(f"   Total Rows: {len(master_df)}")
        return master_df
    else:
        print("\n❌ No valid data found to merge.")
        return None

if __name__ == "__main__":
    merge_csv_datasets(input_folder="..\\data\\patient_content", output_folder="..\\data")