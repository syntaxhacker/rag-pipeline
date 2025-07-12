import pandas as pd
from huggingface_hub import HfApi, HfFolder, create_repo
import os
import json
import getpass

# Function to convert JSON to Parquet and upload to Hugging Face

def json_to_parquet(json_file: str, parquet_file: str, repo_id: str):
    df = pd.read_json(json_file)
    
    # Debugging: Print the first few rows of the DataFrame
    print("DataFrame head:")
    print(df.head())
    
    # Convert dictionaries in 'context' column to JSON strings
    if 'context' in df.columns:
        df['context'] = df['context'].apply(json.dumps)
        print("Converted 'context' column to JSON strings.")
    
    # Convert to Parquet format
    df.to_parquet(parquet_file)
    
    # Upload to Hugging Face
    token = os.getenv("HF_TOKEN") or getpass.getpass("Enter your Hugging Face token: ")
    api = HfApi()
    
    try:
        # Try to create the repository (will do nothing if it already exists)
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
        print(f"Repository {repo_id} is ready.")
        
        # Upload the file
        api.upload_file(
            path_or_fileobj=parquet_file,
            path_in_repo=os.path.basename(parquet_file),
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print(f"Successfully uploaded {parquet_file} to {repo_id}")
    except Exception as e:
        print(f"Error during upload: {str(e)}")
    
if __name__ == '__main__':
    json_file = os.path.join('all_settings_dataset.json')  # Path to your JSON file
    parquet_file = os.path.join('all_settings_dataset.parquet')  # Output Parquet file
    repo_id = 'syntaxhacker/rag_pipeline'  # Hugging Face repo ID
    json_to_parquet(json_file, parquet_file, repo_id) 