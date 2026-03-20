import os
import dropbox
from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()
TOKEN = os.getenv('DROPBOX_TOKEN')
URL = os.getenv('DROPBOX_URL')
PASSWORD = os.getenv('DROPBOX_PASSWORD')

APP_KEY = os.getenv('DROPBOX_APP_KEY')
APP_SECRET = os.getenv('DROPBOX_APP_SECRET')
REFRESH_TOKEN = os.getenv('DROPBOX_REFRESH_TOKEN')

# Initialize Dropbox client
dbx = dropbox.Dropbox(
    app_key=APP_KEY,
    app_secret=APP_SECRET,
    oauth2_refresh_token=REFRESH_TOKEN
)

def download_video_to_temp(patient_id, video_name):
    """
    Downloads a specific video to a temporary folder and returns its local path.
    
    Args:
        patient_id (str): The folder name/ID of the patient.
        video_name (str): The filename of the video (e.g., 'video.mp4').

    Returns:
        str: Absolute path of the downloaded file, or None if an error occurred.
    """
    temp_folder = "../temp_workspace"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
        
    # Create absolute path for local storage
    local_video_path = os.path.abspath(os.path.join(temp_folder, video_name))
    
    # Robust construction of the Dropbox path
    # We assume the structure is /PatientID/VideoName inside the shared link
    dropbox_path = f"/{patient_id}/{video_name}"
    
    print(f"⬇️  Starting download: {video_name} ...")
    
    try:
        # Direct download from the Shared Link
        # This streams the file to memory first, then writes to disk
        metadata, res = dbx.sharing_get_shared_link_file(
            url=URL,
            path=dropbox_path,
            link_password=PASSWORD
        )
        
        with open(local_video_path, "wb") as f:
            f.write(res.content)
            
        size_mb = metadata.size / (1024 * 1024)
        print(f"✅ Download complete: {local_video_path} ({size_mb:.2f} MB)")
        return local_video_path

    except Exception as e:
        print(f"❌ Error downloading {video_name}: {e}")
        # Preventive cleanup if the file was partially created
        if os.path.exists(local_video_path):
            os.remove(local_video_path)
        return None


def delete_temp_video(local_path):
    """
    Deletes the temporary video from the disk.
    It handles PermissionError specifically (common with OpenCV on Windows).
    
    Args:
        local_path (str): The absolute path of the file to delete.
    """
    if not local_path:
        return

    if os.path.exists(local_path):
        try:
            os.remove(local_path)
            print(f"🗑️  File deleted: {os.path.basename(local_path)}")
            return True
        
        except PermissionError:
            print(f"⚠️  PERMISSION DENIED: Could not delete {local_path}.")
            print("    Ensure you called 'cap.release()' in OpenCV before calling this function.")
            return False
            
        except Exception as e:
            print(f"⚠️  Error deleting file: {e}")
            return False
    else:
        print(f"⚠️  File does not exist, skipping deletion: {local_path}")
        return True