import json
import os
import dropbox
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()
TOKEN = os.getenv('DROPBOX_TOKEN')
URL = os.getenv('DROPBOX_URL')
PASSWORD = os.getenv('DROPBOX_PASSWORD')

# Initialize Dropbox client
dbx = dropbox.Dropbox(TOKEN)

def generate_dataset_inventory():
    """
    Crawls the shared Dropbox link and creates a JSON index of all videos and photos.
    This avoids downloading while maintaining a 'map' of the data.
    """
    shared_link = dropbox.files.SharedLink(url=URL, password=PASSWORD)
    dataset_index = []

    print("🚀 Starting the dataset exploration...")
    
    try:
        # List the root folder of the shared link
        res = dbx.files_list_folder(path="", shared_link=shared_link)
        
        for entry in res.entries:
            if isinstance(entry, dropbox.files.FolderMetadata):
                print(f"📁 Processing folder: {entry.name}")
                
                try:
                    # Access subfolders using relative paths (/{folder_name})
                    sub_res = dbx.files_list_folder(path=f"/{entry.name}", shared_link=shared_link)
                    
                    patient_files = []
                    for item in sub_res.entries:
                        if isinstance(item, dropbox.files.FileMetadata):
                            patient_files.append({
                                "name": item.name,
                                "path_lower": item.path_lower,
                                "size_mb": round(item.size / (1024*1024), 2),
                                "extension": item.name.split('.', 1)[-1].lower() if '.' in item.name else "",
                                "is_video": item.name.lower().endswith(('.mp4', '.avi', '.mov'))
                            })
                    
                    dataset_index.append({
                        "patient_id": entry.name,
                        "file_count": len(patient_files),
                        "files": patient_files
                    })
                except Exception as e:
                    print(f"⚠️ Warning: Could not read subfolder {entry.name}: {e}")
                    continue

        # Save the inventory to a JSON file
        with open('dataset_inventory.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_index, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Success! Dataset inventory created with {len(dataset_index)} patient folders.")

    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    generate_dataset_inventory()