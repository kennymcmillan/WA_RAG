import dropbox
import os

# Set up Dropbox API credentials
DROPBOX_APPKEY = os.getenv("DROPBOX_APPKEY")
DROPBOX_APPSECRET = os.getenv("DROPBOX_APPSECRET")
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")

# Set up the Dropbox folder path
DROPBOX_FOLDER_PATH = "/PDF_Files"  # Replace with the actual folder path

def check_dropbox_connection():
    try:
        # Create a Dropbox object with the access token
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

        # Check if the Dropbox folder exists
        try:
            dbx.files_get_metadata(DROPBOX_FOLDER_PATH)
            print(f"Dropbox folder '{DROPBOX_FOLDER_PATH}' exists")
        except dropbox.exceptions.ApiError as e:
            print(f"Error accessing Dropbox folder '{DROPBOX_FOLDER_PATH}': {e}")

        # Check if we can list files in the Dropbox folder
        try:
            files = dbx.files_list_folder(DROPBOX_FOLDER_PATH)
            print(f"Listed files in Dropbox folder '{DROPBOX_FOLDER_PATH}': {files.entries}")
        except dropbox.exceptions.ApiError as e:
            print(f"Error listing files in Dropbox folder '{DROPBOX_FOLDER_PATH}': {e}")

        # Check if we can upload a file to the Dropbox folder
        try:
            file_path = "test_file.txt"
            with open(file_path, "w") as f:
                f.write("Test file contents")
            dbx.files_upload(file_path, DROPBOX_FOLDER_PATH + "/test_file.txt")
            print(f"Uploaded file to Dropbox folder '{DROPBOX_FOLDER_PATH}'")
        except dropbox.exceptions.ApiError as e:
            print(f"Error uploading file to Dropbox folder '{DROPBOX_FOLDER_PATH}': {e}")

    except dropbox.exceptions.AuthError as e:
        print(f"Error authenticating with Dropbox: {e}")

if __name__ == "__main__":
    check_dropbox_connection()