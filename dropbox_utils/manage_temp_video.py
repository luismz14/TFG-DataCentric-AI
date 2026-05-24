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

def download_video_to_temp(patient_id, video_name, verbose=True):
    """
    Downloads a specific video to a temporary folder and returns its local path.

    Args:
        patient_id (str): The folder name/ID of the patient.
        video_name (str): The filename of the video.

    Returns:
        str: Absolute path of the downloaded file, or None if an error occurred.
    """
    temp_folder = "../temp_workspace"
    os.makedirs(temp_folder, exist_ok=True)

    safe_patient_id = str(patient_id).replace(os.sep, "_").replace("/", "_")
    safe_video_name = str(video_name).replace(os.sep, "_").replace("/", "_")
    local_filename = f"{safe_patient_id}__{safe_video_name}"
    local_video_path = os.path.abspath(os.path.join(temp_folder, local_filename))
    partial_video_path = f"{local_video_path}.part"

    if os.path.exists(local_video_path) and os.path.getsize(local_video_path) > 0:
        if verbose:
            print(f"Reusing local video: {local_video_path}")
        return local_video_path

    if os.path.exists(partial_video_path):
        os.remove(partial_video_path)

    dropbox_path = f"/{patient_id}/{video_name}"

    if verbose:
        print(f"Starting download: {video_name} ...")

    try:
        metadata, res = dbx.sharing_get_shared_link_file(
            url=URL,
            path=dropbox_path,
            link_password=PASSWORD,
        )

        with open(partial_video_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        downloaded_size = os.path.getsize(partial_video_path)
        if downloaded_size != metadata.size:
            raise OSError(
                f"Incomplete download for {video_name}: "
                f"{downloaded_size} of {metadata.size} bytes"
            )

        os.replace(partial_video_path, local_video_path)

        if verbose:
            size_mb = metadata.size / (1024 * 1024)
            print(f"Download complete: {local_video_path} ({size_mb:.2f} MB)")

        return local_video_path

    except Exception as e:
        if verbose:
            print(f"Error downloading {video_name}: {e}")

        if os.path.exists(partial_video_path):
            os.remove(partial_video_path)

        if os.path.exists(local_video_path) and os.path.getsize(local_video_path) == 0:
            os.remove(local_video_path)

        return None
    
    
def delete_temp_video(local_path, verbose=True):
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
            if verbose:
                print(f"🗑️  File deleted: {os.path.basename(local_path)}")
            return True
        
        except PermissionError:
            if verbose:
                print(f"⚠️  PERMISSION DENIED: Could not delete {local_path}.")
                print("    Ensure you called 'cap.release()' in OpenCV before calling this function.")
            return False
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Error deleting file: {e}")
            return False
    else:
        if verbose:
            print(f"⚠️  File does not exist, skipping deletion: {local_path}")
        return True
