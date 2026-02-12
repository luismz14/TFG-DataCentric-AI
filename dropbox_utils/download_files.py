import os
import json
import dropbox
from dotenv import load_dotenv

# Cargar configuración
load_dotenv()
TOKEN = os.getenv('DROPBOX_TOKEN')
URL = os.getenv('DROPBOX_URL')
PASSWORD = os.getenv('DROPBOX_PASSWORD')

dbx = dropbox.Dropbox(TOKEN)

def download_metadata_and_images():
    target_extensions = ('.jpg', '.csv', '.json', '.txt')
    base_local_folder = "../data"

    if not os.path.exists('../dataset_inventory.json'):
        print("❌ Error: No se encuentra 'dataset_inventory.json'.")
        return

    with open('../dataset_inventory.json', 'r', encoding='utf-8') as f:
        inventory = json.load(f)

    print(f"🚀 Starting selective download into '{base_local_folder}/'...")

    for patient in inventory:
        patient_id = patient['patient_id']
        local_patient_path = os.path.join(base_local_folder, patient_id)
        
        files_to_download = [f for f in patient['files'] if f['name'].lower().endswith(target_extensions)]

        if files_to_download:
            if not os.path.exists(local_patient_path):
                os.makedirs(local_patient_path)
                print(f"📁 Processing: {patient_id}")

            for file_info in files_to_download:
                file_name = file_info['name']
                local_file_path = os.path.join(local_patient_path, file_name)
                
                dropbox_path = file_info.get('path_lower')
                
                if not dropbox_path:
                    dropbox_path = f"/{patient_id}/{file_name}"
                
                if not dropbox_path.startswith('/'):
                    dropbox_path = f"/{dropbox_path}"

                if not os.path.exists(local_file_path):
                    try:
                        print(f"  ⬇️ Downloading: {file_name}")
                        
                        with open(local_file_path, "wb") as f:
                            metadata, res = dbx.sharing_get_shared_link_file(
                                url=URL,
                                path=dropbox_path,
                                link_password=PASSWORD
                            )
                            f.write(res.content)
                            
                    except Exception as e:
                        print(f"  ⚠️ Error downloading {file_name}: {e}")
                else:
                    print(f"  ✅ Skipping {file_name}")

    print("\n🎉 Selective download complete!")

if __name__ == "__main__":
    download_metadata_and_images()