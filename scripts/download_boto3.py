#!/usr/bin/env python3
"""
Download subjects from OpenNeuro using boto3 directly
No AWS credentials needed - uses anonymous access
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
import sys

DATASET_ID = "ds002718"
BUCKET_NAME = "openneuro.org"

def download_s3_folder(bucket, prefix, local_dir, s3_client):
    """
    Download all files from an S3 prefix to a local directory
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    files_downloaded = 0
    total_size = 0
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            # Get the key (full S3 path)
            key = obj['Key']
            size = obj['Size']
            
            # Skip if it's just a "directory" marker
            if key.endswith('/'):
                continue
            
            # Compute local path
            # Remove the dataset prefix to get relative path
            relative_path = key
            if key.startswith(f"{DATASET_ID}/"):
                relative_path = key[len(f"{DATASET_ID}/"):]
            
            local_file = local_dir / relative_path
            
            # Create parent directories
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download the file
            print(f"  Downloading: {relative_path} ({size / 1024 / 1024:.2f} MB)")
            s3_client.download_file(bucket, key, str(local_file))
            
            files_downloaded += 1
            total_size += size
    
    return files_downloaded, total_size

def main():
    print("=" * 70)
    print(f"OpenNeuro Dataset Downloader (boto3) - {DATASET_ID}")
    print("=" * 70)
    
    # Create S3 client with anonymous access (no credentials needed)
    s3_client = boto3.client(
        's3',
        config=Config(signature_version=UNSIGNED),
        region_name='us-east-1'
    )
    
    # Output directory
    output_dir = Path("data") / DATASET_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Subjects to download
    subjects = ['sub-00', 'sub-00', 'sub-00']
    
    print(f"\nWill download:")
    print(f"  • Dataset metadata files")
    print(f"  • Stimuli (experimental task stimuli)")
    print(f"  • {len(subjects)} subjects:")
    for subj in subjects:
        print(f"    - {subj}")
    
    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    print("\n" + "-" * 70)
    print("Downloading dataset description files...")
    print("-" * 70)
    
    # Download dataset-level files
    dataset_files = [
        'dataset_description.json',
        'README',
        'CHANGES',
        'participants.tsv',
        'participants.json',
        'task-faces_bold.json'
    ]
    
    for filename in dataset_files:
        try:
            s3_key = f"{DATASET_ID}/{filename}"
            local_path = output_dir / filename
            s3_client.download_file(BUCKET_NAME, s3_key, str(local_path))
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  - {filename} (not found or error)")
    
    # Download stimuli folder
    print("\n" + "-" * 70)
    print("Downloading stimuli...")
    print("-" * 70)
    
    stimuli_files = 0
    stimuli_size = 0
    
    try:
        stimuli_prefix = f"{DATASET_ID}/stimuli/"
        stimuli_files, stimuli_size = download_s3_folder(
            BUCKET_NAME,
            stimuli_prefix,
            output_dir,
            s3_client
        )
        if stimuli_files > 0:
            print(f"\n  ✓ Downloaded {stimuli_files} stimulus files ({stimuli_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"  - No stimuli found in dataset")
    except Exception as e:
        print(f"  - Stimuli not available or error: {str(e)}")
    
    # Initialize totals with stimuli
    total_files = stimuli_files
    total_size = stimuli_size
    
    # Download each subject
    print("\n" + "-" * 70)
    print("Downloading subject data...")
    print("-" * 70)
    
    for subject in subjects:
        print(f"\n📦 {subject}:")
        s3_prefix = f"{DATASET_ID}/{subject}/"
        
        try:
            files, size = download_s3_folder(
                BUCKET_NAME,
                s3_prefix,
                output_dir,
                s3_client
            )
            total_files += files
            total_size += size
            print(f"  ✓ Downloaded {files} files ({size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print("\n" + "=" * 70)
    print(f"Download complete!")
    print(f"  Total files: {total_files}")
    print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"  Location: {output_dir.absolute()}")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

