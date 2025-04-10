import os
import dropbox
from dotenv import load_dotenv

def list_all_files():
    """List all files in the Dropbox account"""
    load_dotenv()
    
    # Get access token
    access_token = os.getenv("DROPBOX_TOKEN") or os.getenv("DROPBOX_ACCESS_TOKEN")
    if not access_token:
        print("No Dropbox access token found in .env file")
        return
    
    try:
        # Connect to Dropbox
        dbx = dropbox.Dropbox(access_token)
        print(f"Connected to Dropbox as {dbx.users_get_current_account().name.display_name}")
        
        # List root directory
        print("\nListing root directory:")
        result = dbx.files_list_folder("")
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                print(f"File: {entry.name} (Path: {entry.path_display})")
            elif isinstance(entry, dropbox.files.FolderMetadata):
                print(f"Folder: {entry.name} (Path: {entry.path_display})")
                
                # Try to list contents of this folder
                try:
                    folder_path = entry.path_lower
                    if folder_path.startswith('/'):
                        folder_path = folder_path[1:]  # Remove leading slash
                        
                    print(f"\n  Contents of {entry.name}:")
                    folder_contents = dbx.files_list_folder(folder_path)
                    for item in folder_contents.entries:
                        if isinstance(item, dropbox.files.FileMetadata):
                            print(f"  - File: {item.name}")
                        elif isinstance(item, dropbox.files.FolderMetadata):
                            print(f"  - Subfolder: {item.name}")
                except Exception as e:
                    print(f"  Error listing folder {entry.name}: {str(e)}")
        
        # Try specifically looking for /RAG_documentation folder
        print("\nTrying to access /RAG_documentation folder:")
        try:
            metadata = dbx.files_get_metadata("/RAG_documentation")
            print(f"Found: {metadata.path_display}")
            
            # List contents
            print("Contents:")
            contents = dbx.files_list_folder("RAG_documentation")
            for item in contents.entries:
                print(f" - {item.name} ({'folder' if isinstance(item, dropbox.files.FolderMetadata) else 'file'})")
        except Exception as e:
            print(f"Error accessing /RAG_documentation: {str(e)}")
            
        # Also try Apps folder if it exists
        print("\nTrying to access /Apps folder:")
        try:
            metadata = dbx.files_get_metadata("/Apps")
            print(f"Found: {metadata.path_display}")
            
            # List contents
            print("Contents:")
            contents = dbx.files_list_folder("Apps")
            for item in contents.entries:
                print(f" - {item.name} ({'folder' if isinstance(item, dropbox.files.FolderMetadata) else 'file'})")
                
                # If we find RAG_documentation folder, list its contents
                if item.name == "RAG_documentation" and isinstance(item, dropbox.files.FolderMetadata):
                    try:
                        print("\n  Contents of /Apps/RAG_documentation:")
                        rag_contents = dbx.files_list_folder("Apps/RAG_documentation")
                        for rag_item in rag_contents.entries:
                            print(f"   - {rag_item.name} ({'folder' if isinstance(rag_item, dropbox.files.FolderMetadata) else 'file'})")
                    except Exception as e:
                        print(f"  Error listing /Apps/RAG_documentation: {str(e)}")
        except Exception as e:
            print(f"Error accessing /Apps: {str(e)}")
            
        # Try a simpler approach as last resort using API v2
        try:
            print("\nListing using API v2:")
            result = dbx.files_list_folder_v2("")
            
            print("\nRoot contents:")
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata):
                    print(f"File: {entry.name}")
                elif isinstance(entry, dropbox.files.FolderMetadata):
                    print(f"Folder: {entry.name}")
        except Exception as e:
            print(f"Error listing with API v2: {str(e)}")
            
    except Exception as e:
        print(f"Error connecting to Dropbox: {str(e)}")

if __name__ == "__main__":
    list_all_files() 