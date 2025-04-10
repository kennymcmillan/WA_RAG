import os
import dropbox
from dotenv import load_dotenv

def test_dropbox_connection():
    """Test Dropbox connection using credentials from .env file"""
    load_dotenv()
    
    # Print available credentials (without values)
    print("Available Dropbox environment variables:")
    for var in ["DROPBOX_APPKEY", "DROPBOX_APPSECRET", "DROPBOX_REFRESH_TOKEN", "DROPBOX_TOKEN", "DROPBOX_ACCESS_TOKEN"]:
        if os.getenv(var):
            print(f"✓ {var} is set")
        else:
            print(f"✗ {var} is NOT set")
    
    print("\n--- Trying Direct Access Token Method ---")
    access_token = os.getenv("DROPBOX_TOKEN") or os.getenv("DROPBOX_ACCESS_TOKEN")
    if access_token:
        try:
            # Method 1: Direct access token
            print(f"Attempting to connect with access token...")
            dbx = dropbox.Dropbox(access_token)
            account = dbx.users_get_current_account()
            print(f"Success! Connected as {account.name.display_name}")
            
            # Test specific folder paths
            test_paths = [
                "",  # Root folder
                "Apps",  # Apps folder
                "Apps/RAG_documentation"  # Target folder
            ]
            
            for path in test_paths:
                display_path = f"/{path}" if path else "Root"
                try:
                    print(f"\nTrying to list contents of {display_path} folder:")
                    result = dbx.files_list_folder(path)
                    print(f"✓ Success! Found {len(result.entries)} items:")
                    for entry in result.entries:
                        type_str = "folder" if isinstance(entry, dropbox.files.FolderMetadata) else "file"
                        print(f"  - {entry.name} ({type_str})")
                except Exception as e:
                    print(f"✗ Error listing {display_path}: {str(e)}")
            
            # Try using get_metadata on our target folder
            test_metadata_paths = [
                "/Apps",
                "/Apps/RAG_documentation"
            ]
            
            for path in test_metadata_paths:
                try:
                    print(f"\nChecking if path exists: {path}")
                    metadata = dbx.files_get_metadata(path)
                    print(f"✓ Path exists! Type: {'folder' if isinstance(metadata, dropbox.files.FolderMetadata) else 'file'}")
                except Exception as e:
                    print(f"✗ Path doesn't exist or error: {str(e)}")
            
            return dbx
        except Exception as e:
            print(f"Error with direct token method: {str(e)}")
    else:
        print("No access token available to try direct method")
    
    print("\n--- Trying OAuth2 Refresh Token Method ---")
    app_key = os.getenv("DROPBOX_APPKEY")
    app_secret = os.getenv("DROPBOX_APPSECRET")
    refresh_token = os.getenv("DROPBOX_REFRESH_TOKEN")
    
    if app_key and app_secret and refresh_token:
        try:
            # Method 2: OAuth2 with refresh token
            print(f"Attempting to connect with OAuth2 refresh token...")
            dbx = dropbox.Dropbox(
                app_key=app_key,
                app_secret=app_secret,
                oauth2_refresh_token=refresh_token
            )
            account = dbx.users_get_current_account()
            print(f"Success! Connected as {account.name.display_name}")
            
            # Try listing root folder
            print("Listing root folder...")
            result = dbx.files_list_folder("")
            print(f"Found {len(result.entries)} items in root folder")
            for entry in result.entries[:5]:  # Show first 5 items
                print(f" - {entry.name} ({'folder' if isinstance(entry, dropbox.files.FolderMetadata) else 'file'})")
            
            return dbx
        except Exception as e:
            print(f"Error with OAuth2 method: {str(e)}")
    else:
        print("Missing required OAuth2 credentials")
    
    print("\nNo successful connection methods. Please run setup_dropbox.py to generate new credentials.")
    return None

if __name__ == "__main__":
    dbx = test_dropbox_connection()
    
    if dbx:
        # Try to create and access the RAG_documentation folder
        folder_path = "/RAG_documentation"
        print(f"\nTrying to access folder: {folder_path}")
        
        # Check if folder exists
        try:
            dbx.files_get_metadata(folder_path)
            print(f"✓ Folder {folder_path} exists")
        except:
            print(f"✗ Folder {folder_path} does not exist")
            print(f"Attempting to create folder {folder_path}...")
            try:
                dbx.files_create_folder_v2(folder_path)
                print(f"✓ Successfully created folder {folder_path}")
            except Exception as e:
                print(f"✗ Failed to create folder: {str(e)}") 