import pandas as pd
import shutil
from pathlib import Path

def unifyExcels():
    base_path = Path('../data/patient_content')
    output_path = Path('../data/unified_data.csv')
    all_dfs = []
    
    for patient_folder in (f for f in base_path.iterdir() if f.is_dir()):
        for csv_file in patient_folder.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                if df.empty:
                    continue

                raw_strings = df.iloc[:, 0].astype(str)
                split_data = raw_strings.str.split('_', expand=True)

                if split_data.shape[1] < 4:
                    print(f'Invalid format at {csv_file.name}')
                    continue

                temp_df = pd.DataFrame({
                    'patient_id': patient_folder.name,
                    'day': split_data[0],
                    'hour': split_data[1],
                    'R': split_data[2],
                    'F': split_data[3],
                    'histology': df.get('assumed histology', 'N/A'),
                    'filename': raw_strings
                })
                all_dfs.append(temp_df)
                
            except Exception as e:
                print(f'Error processing {csv_file.name}: {e}')

    if not all_dfs:
        print("No data found to unify.")
        return

    fulldf = pd.concat(all_dfs, ignore_index=True)

    # Histology cleaning like agreed with the tutor.
    fulldf['histology'] = (
        fulldf['histology']
        .str.strip()
        .str.replace('Sessile serrated adenoma (with displasia)', 'Sessile serrated adenoma', regex=False)
        .str.replace('Sessile serrated adenoma', 'Sessile_serrated_adenoma', regex=False)
    )
    fulldf = fulldf[fulldf['histology'] != 'Lipoma']

    fulldf.to_csv(output_path, index=False)
    print(f'CSV unificado en: {output_path}')

def unifyImages():
    src_base = Path('../data/patient_content')
    dst_dir = Path('../data/unified_images')

    dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for patient_folder in (f for f in src_base.iterdir() if f.is_dir()):
        for img_path in patient_folder.glob('*.jpg'):
            dst_path = dst_dir / img_path.name
            try:
                shutil.copy2(img_path, dst_path)
                count += 1
            except Exception as e:
                print(f'Error copying {img_path.name}: {e}')
    
    print(f'Process completed. {count} files copied to {dst_dir}')

if __name__ == '__main__':
    unifyExcels()
    unifyImages()