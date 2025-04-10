import os
import dropbox
from dotenv import load_dotenv

def list_root_files():
    """List all files at the root level of Dropbox"""
    load_dotenv()
    
    # Get access token
    access_token = os.getenv("DROPBOX_TOKEN")
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
        
        print(f"Found {len(result.entries)} items at root level:")
        
        # Track the PDFs we find
        pdf_files = []
        all_files = []
        folders = []
        
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                file_type = "PDF" if entry.name.lower().endswith('.pdf') else "File"
                print(f"{file_type}: {entry.name} (Path: {entry.path_display})")
                all_files.append(entry)
                if entry.name.lower().endswith('.pdf'):
                    pdf_files.append(entry)
            else:
                print(f"Folder: {entry.name} (Path: {entry.path_display})")
                folders.append(entry)
                
        # Output summary of PDFs
        if pdf_files:
            print(f"\nFound {len(pdf_files)} PDF files at root level:")
            for pdf in pdf_files:
                print(f" - {pdf.name} (Size: {pdf.size} bytes)")
                
            # Look for Drake file specifically
            drake_files = [f for f in pdf_files if "drake" in f.name.lower()]
            if drake_files:
                print("\nFound Drake PDF files:")
                for drake_file in drake_files:
                    print(f" - {drake_file.name}")
                    
                    # Try to download this file
                    print(f"\nAttempting to download {drake_file.name}...")
                    try:
                        metadata, res = dbx.files_download(path=drake_file.path_display)
                        print(f"Successfully downloaded {metadata.name} ({len(res.content)} bytes)")
                    except Exception as download_error:
                        print(f"Error downloading file: {str(download_error)}")
            else:
                print("\nNo files with 'Drake' in the name found at root level")
        else:
            print("\nNo PDF files found at root level")
            
        # If we have any folders, try to list RAG_documentation if it exists
        rag_folder = next((f for f in folders if f.name.lower() == "rag_documentation"), None)
        if rag_folder:
            print(f"\nFound RAG_documentation folder: {rag_folder.path_display}")
            try:
                print("Listing contents of RAG_documentation folder:")
                rag_contents = dbx.files_list_folder(rag_folder.path_lower)
                for item in rag_contents.entries:
                    item_type = "File" if isinstance(item, dropbox.files.FileMetadata) else "Folder"
                    print(f" - {item_type}: {item.name}")
            except Exception as folder_error:
                print(f"Error listing RAG_documentation folder: {str(folder_error)}")
        
        # Try to search specifically for Drake files
        try:
            print("\nSearching for Drake files using search API...")
            search_results = dbx.files_search_v2("Drake")
            print(f"Search found {len(search_results.matches)} matches")
            
            for match in search_results.matches:
                metadata = match.metadata
                if isinstance(metadata, dropbox.files.FileMetadata):
                    print(f" - File: {metadata.path_display}")
                else:
                    print(f" - Folder: {metadata.path_display}")
        except Exception as search_error:
            print(f"Error searching: {str(search_error)}")
            
            # Try legacy search
            try:
                print("Trying legacy search...")
                search_results = dbx.files_search("Drake")
                print(f"Legacy search found {len(search_results.matches)} matches")
                
                for match in search_results.matches:
                    metadata = match.metadata
                    print(f" - {metadata.path_display}")
            except Exception as legacy_error:
                print(f"Legacy search error: {str(legacy_error)}")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    list_root_files() 