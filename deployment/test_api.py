#!/usr/bin/env python3
"""
Test script for the WSI Processing API with the updated directory structure
"""
import os
import time
import json
import requests
import argparse
import sys

def print_colored(text, color="white"):
    """Print colored text to terminal"""
    colors = {
        "white": "\033[0m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m"
    }
    
    print(f"{colors.get(color, colors['white'])}{text}{colors['white']}")

def test_api(api_url, file_path, model_id=None, chunk_size=512, num_workers=4):
    """Test the WSI Processing API workflow"""
    # 1. Check API health
    print_colored("\nChecking API health...", "yellow")
    try:
        response = requests.get(f"{api_url}/health")
        response.raise_for_status()  # Raise exception for non-200 response
        health_data = response.json()
        print_colored("API is healthy!", "green")
        print(f"Current model: {health_data.get('current_model')}")
        print(f"Models available: {health_data.get('models_available')}")
    except Exception as e:
        print_colored(f"API health check failed: {str(e)}", "red")
        return False
    
    # 2. List available models
    print_colored("\nListing available models...", "yellow")
    try:
        response = requests.get(f"{api_url}/models")
        response.raise_for_status()
        models = response.json()
        print(f"Available models:")
        for i, model in enumerate(models.get('models', [])):
            print(f"  {i+1}. {model['name']} ({model['id']})")
        
        # If model_id is specified, verify it exists
        if model_id:
            model_exists = any(model['id'] == model_id for model in models.get('models', []))
            if not model_exists:
                print_colored(f"Specified model ID '{model_id}' not found!", "red")
                return False
    except Exception as e:
        print_colored(f"Failed to list models: {str(e)}", "red")
        return False
    
    # 3. Upload file and start processing
    print_colored("\nUploading file for processing...", "yellow")
    try:
        # Prepare form data with optional model_id
        form_data = {
            'chunk_size': str(chunk_size),
            'num_workers': str(num_workers)
        }
        
        if model_id:
            form_data['model_id'] = model_id
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(f"{api_url}/upload", files=files, data=form_data)
        
        response.raise_for_status()
        result = response.json()
        job_id = result.get('job_id')
        
        if not job_id:
            print_colored("Failed to get job ID from response", "red")
            print(json.dumps(result, indent=2))
            return False
            
        print_colored(f"Processing started with job ID: {job_id}", "green")
    except Exception as e:
        print_colored(f"Upload failed: {str(e)}", "red")
        return False
    
    # 4. Poll for status
    print_colored("\nPolling for job status...", "yellow")
    status = "running"
    progress = 0
    
    while status in ["running", "idle"]:
        try:
            response = requests.get(f"{api_url}/status/{job_id}")
            response.raise_for_status()
            status_data = response.json()
            
            status = status_data.get('status', '')
            new_progress = status_data.get('progress', 0)
            message = status_data.get('message', '')
            
            # Only print if progress changed
            if new_progress > progress:
                progress = new_progress
                print(f"Progress: {progress}% - {message}")
            
            time.sleep(5)
        except Exception as e:
            print_colored(f"Status check failed: {str(e)}", "red")
            return False
    
    # 5. Final status
    if status == "complete":
        print_colored("\nProcessing completed successfully!", "green")
        
        # Download the result
        print_colored("\nDownloading result...", "yellow")
        try:
            output_path = f"output_{job_id}.tiff"
            response = requests.get(f"{api_url}/result/{job_id}", stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print_colored(f"Result saved as {output_path}", "green")
            return True
        except Exception as e:
            print_colored(f"Download failed: {str(e)}", "red")
            return False
    else:
        print_colored(f"\nProcessing failed with status: {status}", "red")
        print(json.dumps(status_data, indent=2))
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the WSI Processing API")
    parser.add_argument("file_path", help="Path to WSI file (SVS or TIFF)")
    parser.add_argument("--api-url", default="http://localhost:5000/api", help="API base URL")
    parser.add_argument("--model-id", help="Specific model ID to use")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for processing")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print_colored("WSI Processing API Test Script", "blue")
    print_colored("===============================")
    print(f"Testing directory structure with files in deployment/")
    
    success = test_api(
        args.api_url,
        args.file_path,
        model_id=args.model_id,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers
    )
    
    if success:
        print_colored("\nTest completed successfully!", "green")
        exit(0)
    else:
        print_colored("\nTest failed!", "red")
        exit(1)