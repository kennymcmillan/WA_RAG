"""
app_dropbox.py

This module manages Dropbox integration for document access.
Provides:
- Authentication with Dropbox API using refresh tokens
- Listing folders and PDF files from Dropbox
- Downloading files from Dropbox
- Uploading files to Dropbox
- Converting downloaded files to file-like objects for processing
"""

import os
import dropbox
from dropbox.exceptions import AuthError as DropboxOAuth2TokenError
import logging
import streamlit as st
import io
from dotenv import load_dotenv
import time

# Load environment variables to ensure we have access to all Dropbox credentials
load_dotenv()

# Get Dropbox credentials from environment variables
def get_dropbox_credentials():
    """Get Dropbox credentials from environment variables"""
    credentials = {
        "app_key": os.getenv("DROPBOX_APPKEY"),
        "app_secret": os.getenv("DROPBOX_APPSECRET"),
        "refresh_token": os.getenv("DROPBOX_REFRESH_TOKEN"),
        "access_token": os.getenv("DROPBOX_TOKEN")
    }
    
    # Log available credentials (without showing the actual values)
    available = [k for k, v in credentials.items() if v]
    logging.info(f"Available Dropbox credentials: {available}")
    
    return credentials

# Initialize Dropbox client with proper token refresh
def get_dropbox_client():
    """
    Initialize Dropbox client with token refresh capability.
    Uses access token first (simpler method), falls back to refresh token.
    """
    credentials = get_dropbox_credentials()
    
    # Try using access token first (simpler method)
    if credentials["access_token"]:
        try:
            logging.info("Initializing Dropbox client with access token")
            dbx = dropbox.Dropbox(credentials["access_token"])
            # Test the connection
            dbx.users_get_current_account()
            logging.info("Successfully authenticated with Dropbox using access token")
            return dbx
        except DropboxOAuth2TokenError as e:
            logging.error(f"Dropbox token expired or invalid (access token method): {str(e)}")
            logging.info("Attempting to refresh token...") # Added logging
        except Exception as e:
            logging.error(f"Error initializing Dropbox with access token: {str(e)}")

    logging.info("Checking for refresh token credentials...") # Added logging - check if we reach this point
    # Fall back to refresh token if access token not available or failed
    if credentials["refresh_token"] and credentials["app_key"] and credentials["app_secret"]:
        logging.info("Refresh token, app key, and app secret are available.") # Added logging
        try:
            logging.info("Initializing Dropbox client with refresh token")
            dbx = dropbox.Dropbox(
                app_key=credentials["app_key"],
                app_secret=credentials["app_secret"],
                oauth2_refresh_token=credentials["refresh_token"]
            )
            # Test the connection
            dbx.users_get_current_account()
            logging.info("Successfully authenticated with Dropbox using refresh token")
            return dbx
        except Exception as e:
            logging.error(f"Error initializing Dropbox with refresh token: {str(e)}")
    else:
        logging.warning("Refresh token, app key, or app secret is missing.") # Added logging
    
    # If we get here, neither method worked
    logging.error("Failed to authenticate with Dropbox")
    return None

# List folders in Dropbox
def list_dropbox_folders(dbx=None):
    if not dbx:
        dbx = get_dropbox_client()
    if not dbx:
        return []
    
    try:
        folders = ["/"]  # Keep "/" for display purposes
        # Use path="" for root listing as per API requirement
        result = dbx.files_list_folder(path="")
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FolderMetadata):
                folders.append(entry.path_display)
        return folders
    except Exception as e:
        logging.error(f"Error listing Dropbox folders: {str(e)}")
        st.error(f"Error listing Dropbox folders: {str(e)}")
        return []

# List PDF files in a Dropbox folder
def list_dropbox_pdf_files(folder_path, dbx=None):
    if not dbx:
        dbx = get_dropbox_client()
    if not dbx:
        return []
    
    try:
        # Always list from root level regardless of requested folder
        logging.info("Listing PDF files from Dropbox root folder")
        
        # List files in the root folder (use empty string for root as per API requirements)
        result = dbx.files_list_folder("")  # Empty string for root folder
        files = []
        
        # Log all entries found
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                if entry.name.lower().endswith('.pdf'):
                    logging.info(f"Found PDF in root: {entry.name}")
                    files.append(entry.name)
                else:
                    logging.info(f"Found non-PDF file in root: {entry.name}")
            elif isinstance(entry, dropbox.files.FolderMetadata):
                logging.info(f"Found folder in root: {entry.name}")
        
        # Check specifically for Drake file
        drake_files = [f for f in files if 'drake' in f.lower()]
        if drake_files:
            logging.info(f"Found Drake file(s): {drake_files}")
        
        logging.info(f"Found {len(files)} PDF files in root folder")
        return files
    except Exception as e:
        logging.error(f"Error listing files in Dropbox: {str(e)}")
        st.error(f"Error listing Dropbox files: {str(e)}")
        return []

# Download a file from Dropbox
def download_dropbox_file(file_path, dbx=None):
    if not dbx:
        dbx = get_dropbox_client()
    if not dbx:
        return None
    
    try:
        # For files_download: Must have leading slash
        if not file_path.startswith("/"):
            api_file_path = f"/{file_path}"
        else:
            api_file_path = file_path
            
        logging.info(f"Downloading Dropbox file: {api_file_path}")
        
        try:
            md, res = dbx.files_download(path=api_file_path)
            data = res.content
            logging.info(f"Successfully downloaded {api_file_path}")
            return data
        except Exception as download_error:
            logging.error(f"Download error: {str(download_error)}")
            return None
    except Exception as e:
        logging.error(f"Error downloading Dropbox file: {str(e)}")
        st.error(f"Error downloading Dropbox file: {str(e)}")
        return None

# Upload a file to Dropbox
def upload_to_dropbox(file_data, file_name, folder_path="/", dbx=None):
    if not dbx:
        dbx = get_dropbox_client()
    if not dbx:
        return False
    
    try:
        # Always upload to root level 
        logging.info(f"Uploading file {file_name} to root level in Dropbox")
        
        # Upload the file to root with leading slash for path
        file_path = f"/{file_name}"
        logging.info(f"Uploading file to Dropbox: {file_path}")
        
        dbx.files_upload(
            file_data, 
            file_path,  # Use root path with leading slash
            mode=dropbox.files.WriteMode.overwrite
        )
        logging.info(f"Successfully uploaded {file_name} to Dropbox root")
        return True
    except Exception as e:
        logging.error(f"Error uploading to Dropbox: {str(e)}")
        st.error(f"Error uploading to Dropbox: {str(e)}")
        return False

# Create a file-like object from Dropbox file data
def create_file_like_object(file_data, file_name):
    if not file_data:
        return None
    
    try:
        file_like = io.BytesIO(file_data)
        file_like.name = file_name  # Add name attribute
        return file_like
    except Exception as e:
        logging.error(f"Error creating file-like object: {str(e)}")
        return None

# Check if Dropbox is configured
def is_dropbox_configured():
    """Check if Dropbox credentials are properly configured"""
    credentials = get_dropbox_credentials()
    if credentials["app_key"] and credentials["app_secret"] and credentials["refresh_token"]:
        return True
    return False

# Save a file to Dropbox
def save_file_to_dropbox(file_obj, path):
    """
    Save a file to Dropbox
    
    Args:
        file_obj: File object (must have read method)
        path: Dropbox path where the file should be saved
        
    Returns:
        bool: Success indicator
    """
    try:
        # Get Dropbox client
        dbx = get_dropbox_client()
        if not dbx:
            logging.error("Could not initialize Dropbox client")
            return False
        
        # Read file data
        file_data = file_obj.read()
        
        # Extract filename from path
        filename = os.path.basename(path)
        folder_path = os.path.dirname(path)
        if not folder_path:
            folder_path = "/"
            
        # Upload to Dropbox
        return upload_to_dropbox(file_data, filename, folder_path, dbx)
    except Exception as e:
        logging.error(f"Error saving file to Dropbox: {str(e)}")
        return False

# Helper function to set up Dropbox using OAuth flow (call this manually to get tokens)
def setup_dropbox_oauth():
    """
    Run this function in a Python script to set up Dropbox OAuth.
    You'll need to copy and paste the authorization code from the browser.
    Then update your .env file with the tokens.
    """
    from dropbox import DropboxOAuth2FlowNoRedirect
    
    app_key = input("Enter your Dropbox app key: ").strip()
    app_secret = input("Enter your Dropbox app secret: ").strip()
    
    auth_flow = DropboxOAuth2FlowNoRedirect(
        app_key,
        consumer_secret=app_secret,
        token_access_type='offline',
        scope=['files.metadata.read', 'files.content.read', 'files.content.write', 'account_info.read']
    )
    
    authorize_url = auth_flow.start()
    print("1. Go to: " + authorize_url)
    print("2. Click \"Allow\" (you might have to log in first).")
    print("3. Copy the authorization code.")
    auth_code = input("Enter the authorization code here: ").strip()
    
    try:
        oauth_result = auth_flow.finish(auth_code)
        print("\nDROPBOX_APPKEY =", app_key)
        print("DROPBOX_APPSECRET =", app_secret)
        print("DROPBOX_REFRESH_TOKEN =", oauth_result.refresh_token)
        print("DROPBOX_TOKEN =", oauth_result.access_token)
        print("\nScopes:", oauth_result.scope)
        print("Token expires at:", oauth_result.expires_at)
        print("\nCopy these values to your .env file")
        return oauth_result
    except Exception as e:
        print('Error: %s' % (e,))
        return None
