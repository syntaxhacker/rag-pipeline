import pandas as pd
from huggingface_hub import HfApi, create_repo
import os
import json
import getpass

def upload_developer_portfolio():
    # Read the JSON file
    df = pd.read_json('developer_portfolio_dataset.json')
    
    # Display info about the dataset
    print("DataFrame head:")
    print(df.head())
    print(f"Dataset shape: {df.shape}")
    
    # Convert context column to JSON strings if it exists
    if 'context' in df.columns:
        df['context'] = df['context'].apply(json.dumps)
        print("Converted 'context' column to JSON strings.")
    
    # Create parquet file
    parquet_file = 'developer_portfolio_dataset.parquet'
    df.to_parquet(parquet_file)
    print(f"Created parquet file: {parquet_file}")
    
    # Upload to Hugging Face
    token = os.getenv("HF_TOKEN") or getpass.getpass("Enter your Hugging Face token: ")
    repo_id = 'syntaxhacker/developer-portfolio-rag'  # New repo for developer portfolio
    
    api = HfApi()
    
    try:
        # Create the repository
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
        print(f"Repository {repo_id} is ready.")
        
        # Upload the parquet file
        api.upload_file(
            path_or_fileobj=parquet_file,
            path_in_repo=parquet_file,
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print(f"Successfully uploaded {parquet_file} to {repo_id}")
        
        # Also upload the JSON file for reference
        api.upload_file(
            path_or_fileobj='developer_portfolio_dataset.json',
            path_in_repo='developer_portfolio_dataset.json',
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print(f"Successfully uploaded JSON file to {repo_id}")
        
    except Exception as e:
        print(f"Error during upload: {str(e)}")

if __name__ == '__main__':
    upload_developer_portfolio()