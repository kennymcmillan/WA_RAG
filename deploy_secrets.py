#!/usr/bin/env python
"""
deploy_secrets.py

Utility script to convert .env file variables to the Streamlit Cloud secrets format.
This script focuses on generating output specifically formatted for the Streamlit Cloud deployment.
"""

import os
from dotenv import dotenv_values

def deploy_secrets():
    """Convert .env variables to format ready for Streamlit Cloud."""
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Error: .env file not found in the current directory.")
        return False
    
    # Read all variables from .env file
    env_vars = dotenv_values('.env')
    
    if not env_vars:
        print("Warning: No variables found in .env file.")
        return False
    
    # Create sections for organizing the secrets
    sections = {
        "database": ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", 
                    "DATABASE_NAME", "DATABASE_USER", "DATABASE_PASSWORD", "DATABASE_HOST", "DATABASE_PORT"],
        "dropbox": ["DROPBOX_APPKEY", "DROPBOX_APPSECRET", "DROPBOX_REFRESH_TOKEN", "DROPBOX_TOKEN"],
        "openrouter": ["OPENROUTER_API_KEY"]
    }
    
    # Initialize the TOML content
    toml_content = ""
    
    # Process each section
    for section, keys in sections.items():
        section_vars = {}
        for key in keys:
            if key in env_vars and env_vars[key]:
                # Handle DATABASE_ prefix conversion
                if key.startswith("DATABASE_"):
                    alt_key = "DB_" + key[9:]  # Remove DATABASE_ prefix
                    if alt_key not in env_vars:  # Only use DATABASE_ version if DB_ isn't set
                        section_vars[alt_key] = env_vars[key]
                else:
                    section_vars[key] = env_vars[key]
        
        # Add section to TOML if it has any variables
        if section_vars:
            toml_content += f"[{section}]\n"
            for key, value in section_vars.items():
                toml_content += f'{key} = "{value}"\n'
            toml_content += "\n"
    
    # Add any remaining variables not in predefined sections
    other_vars = {}
    for key, value in env_vars.items():
        if not any(key in section_keys for section_keys in sections.values()):
            other_vars[key] = value
    
    if other_vars:
        toml_content += "[general]\n"
        for key, value in other_vars.items():
            toml_content += f'{key} = "{value}"\n'
    
    print("\n=== STREAMLIT CLOUD SECRETS ===\n")
    print("Copy everything between the lines and paste into Streamlit Cloud's secrets manager:\n")
    print("-------- START COPYING BELOW THIS LINE --------")
    print(toml_content.strip())
    print("-------- STOP COPYING ABOVE THIS LINE --------\n")
    
    print("Instructions for Streamlit Cloud:")
    print("1. Go to your app's dashboard: https://share.streamlit.io/")
    print("2. Select your app")
    print("3. Click on 'Settings' ⚙️ > 'Secrets'")
    print("4. Paste the above content into the text area")
    print("5. Click 'Save'")
    print("\nYour app will automatically use these secrets when deployed on Streamlit Cloud.\n")

if __name__ == "__main__":
    deploy_secrets() 