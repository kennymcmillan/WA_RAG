"""
setup_dropbox.py

This script helps you set up Dropbox integration for the Aspire Academy Document Assistant.
It guides you through the OAuth process to obtain the necessary tokens.

Instructions:
1. Run this script: python setup_dropbox.py
2. Follow the prompts to enter your Dropbox app key and secret
3. Open the provided URL in a browser and authorize the application
4. Copy the authorization code and paste it back in the terminal
5. Copy the generated tokens to your .env file
"""

import os
from dotenv import load_dotenv, set_key
from app_dropbox import setup_dropbox_oauth
from dropbox.exceptions import AuthError as DropboxOAuth2TokenError

def main():
    print("\n=== Aspire Academy Document Assistant - Dropbox Setup ===\n")
    print("This script will help you set up Dropbox integration.")
    print("You'll need to create a Dropbox app first if you haven't already.\n")
    print("1. Go to https://www.dropbox.com/developers/apps")
    print("2. Click 'Create app'")
    print("3. Choose 'Scoped access' and 'Full Dropbox' access")
    print("4. Give your app a name (e.g., 'Aspire Document Assistant')")
    print("5. Once created, find your App key and App secret")
    print("\nAfter completing this setup, tokens will be automatically added to your .env file.\n")
    
    # Check if we already have tokens in .env
    load_dotenv()
    has_app_key = bool(os.getenv("DROPBOX_APPKEY"))
    has_app_secret = bool(os.getenv("DROPBOX_APPSECRET"))
    has_refresh_token = bool(os.getenv("DROPBOX_REFRESH_TOKEN"))
    
    if has_app_key and has_app_secret and has_refresh_token:
        print("You already have Dropbox credentials in your .env file.")
        proceed = input("Do you want to generate new tokens anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("\nSetup cancelled. Your existing tokens will continue to be used.")
            return
    
    # Start the OAuth flow
    print("\nStarting Dropbox OAuth flow...\n")
    result = setup_dropbox_oauth()
    
    if result:
        # Get the app key and secret from user input, since they're not included in the OAuth result
        app_key = input("\nPlease enter the Dropbox app key again for confirmation: ").strip()
        app_secret = input("Please enter the Dropbox app secret again for confirmation: ").strip()
        
        # Check if .env file exists
        env_path = ".env"
        if not os.path.exists(env_path):
            # Create .env file if it doesn't exist
            with open(env_path, "w") as f:
                f.write("# Dropbox integration\n")
        
        # Update .env file with the new tokens
        set_key(env_path, "DROPBOX_APPKEY", app_key)
        set_key(env_path, "DROPBOX_APPSECRET", app_secret)
        set_key(env_path, "DROPBOX_REFRESH_TOKEN", result.refresh_token)
        set_key(env_path, "DROPBOX_TOKEN", result.access_token)
        
        print("\nSetup completed successfully!")
        print("Your .env file has been updated with the new Dropbox credentials.")
        
        # Display tokens for confirmation
        print("\nYour Dropbox tokens:")
        print(f"DROPBOX_APPKEY = {app_key}")
        print(f"DROPBOX_APPSECRET = {app_secret}")
        print(f"DROPBOX_REFRESH_TOKEN = {result.refresh_token}")
        print(f"DROPBOX_TOKEN = {result.access_token}")
    else:
        print("\nSetup failed. Please try again.")

if __name__ == "__main__":
    main() 