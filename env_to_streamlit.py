#!/usr/bin/env python
"""
env_to_streamlit.py

Utility script to convert .env file variables to Streamlit secrets.toml format.
"""

import os
import sys
from dotenv import dotenv_values

def env_to_streamlit_secrets():
    """Convert .env variables to Streamlit secrets.toml format."""
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Error: .env file not found in the current directory.")
        return False
    
    # Read all variables from .env file
    config = dotenv_values('.env')
    
    if not config:
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
    toml_content = "# .streamlit/secrets.toml\n\n"
    
    # Process each section
    for section, keys in sections.items():
        section_vars = {}
        for key in keys:
            if key in config and config[key]:
                # Convert DATABASE_ prefix to DB_ if needed for consistency
                if key.startswith("DATABASE_"):
                    alt_key = "DB_" + key[9:]  # Remove "DATABASE_" and add "DB_"
                    if alt_key not in config:  # Only use the DATABASE_ version if DB_ isn't set
                        section_vars[alt_key] = config[key]
                else:
                    section_vars[key] = config[key]
        
        # Add section to TOML if it has any variables
        if section_vars:
            toml_content += f"[{section}]\n"
            for key, value in section_vars.items():
                toml_content += f'{key} = "{value}"\n'
            toml_content += "\n"
    
    # Add any remaining variables not in predefined sections
    other_vars = {}
    for key, value in config.items():
        if not any(key in section_keys for section_keys in sections.values()):
            other_vars[key] = value
    
    if other_vars:
        toml_content += "[general]\n"
        for key, value in other_vars.items():
            toml_content += f'{key} = "{value}"\n'
    
    # Print instructions
    print("\n=== Streamlit Secrets.toml Content ===\n")
    print(toml_content)
    print("\n=== Instructions ===\n")
    print("1. In your Streamlit Cloud dashboard, navigate to your app's settings")
    print("2. Go to the 'Secrets' section")
    print("3. Copy and paste the above content into the secrets manager")
    print("4. Save your changes")
    print("\nOR for local development:")
    print("1. Create a .streamlit directory in your project root if it doesn't exist")
    print("2. Create a file named 'secrets.toml' inside that directory")
    print("3. Copy and paste the above content into that file")
    print("4. Make sure to add .streamlit/secrets.toml to your .gitignore file\n")
    
    # Offer to save the file
    save = input("Would you like to save this as .streamlit/secrets.toml? (y/n): ").lower()
    if save == 'y':
        # Create .streamlit directory if it doesn't exist
        os.makedirs('.streamlit', exist_ok=True)
        
        # Write the secrets.toml file
        with open('.streamlit/secrets.toml', 'w') as f:
            f.write(toml_content)
        
        # Add to .gitignore if not already there
        add_to_gitignore = True
        if os.path.exists('.gitignore'):
            with open('.gitignore', 'r') as f:
                if '.streamlit/secrets.toml' in f.read():
                    add_to_gitignore = False
        
        if add_to_gitignore:
            with open('.gitignore', 'a') as f:
                f.write('\n# Streamlit secrets\n.streamlit/secrets.toml\n')
            print("Added .streamlit/secrets.toml to .gitignore")
        
        print("Saved to .streamlit/secrets.toml")
    
    return True

if __name__ == "__main__":
    env_to_streamlit_secrets() 