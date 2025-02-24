import os
import subprocess
import threading
import time
import json
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessingController:
    """
    Controller for the Diffusion-FFPE image processing pipeline.
    Handles preprocessing, inference, and restitching steps.
    """
    def __init__(self, model_config=None, output_dir="./results"):
        """
        Initialize the processing controller.
        
        Args:
            model_config: Dictionary containing model configuration
                {
                    "id": "unique-model-id",
                    "name": "Model Name",
                    "checkpoint_path": "./checkpoints/model.pkl",
                    "script_path": "inference_ddp.py",
                    "prompt": "paraffin section",
                    "type": "diffusion",
                    "params": {"additional": "parameters"}
                }
            output_dir: Base directory for outputs
        """
        # Default model config if none provided
        self.model_config = model_config or {
            "id": "diffusion-ffpe",
            "name": "Diffusion FFPE",
            "checkpoint_path": os.path.join(project_root, "checkpoints/model.pkl"),
            "script_path": os.path.join(project_root, "app/inference_ddp.py"),
            "prompt": "paraffin section",
            "type": "diffusion",
            "params": {}
        }
        
        # For backwards compatibility
        self.model_checkpoint_path = self.model_config.get("checkpoint_path")
        self.prompt = self.model_config.get("prompt")
        
        self.gpu_count = 1
        self.base_output_dir = output_dir
        self.status = {
            "status": "idle",
            "progress": 0,
            "message": "Ready to process",
            "current_step": None,
            "job_id": None,
            "error": None
        }
        self._lock = threading.Lock()
        self._current_process = None
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"ProcessingController initialized with model: {self.model_config['name']}")
    
    def update_status(self, **kwargs):
        """Update the status dictionary with thread safety"""
        with self._lock:
            self.status.update(kwargs)
            return self.status.copy()
    
    def get_status(self):
        """Get a copy of the current status"""
        with self._lock:
            return self.status.copy()
            
    def set_model_config(self, model_config):
        """
        Update the model configuration
        
        Args:
            model_config: Dictionary with new model configuration
        """
        with self._lock:
            self.model_config = model_config
            # Update backward compatibility attributes
            self.model_checkpoint_path = model_config.get("checkpoint_path")
            self.prompt = model_config.get("prompt")
            logger.info(f"Model configuration updated to: {model_config['name']}")
            return self.model_config.copy()
    
    def get_model_config(self):
        """Get a copy of the current model configuration"""
        with self._lock:
            return self.model_config.copy()
    
    def process_image(self, input_file_path, job_id=None, chunk_size=512, num_workers=None):
        """
        Process a whole slide image through the entire pipeline.
        
        Args:
            input_file_path: Path to the input SVS or TIFF file
            job_id: Unique identifier for this processing job
            chunk_size: Size of the chunks for processing
            num_workers: Number of workers for parallel processing
            
        Returns:
            job_id: The job identifier for tracking progress
        """
        if job_id is None:
            job_id = f"job_{int(time.time())}"
        
        # Create job-specific directories
        job_dir = os.path.join(self.base_output_dir, job_id)
        chunks_dir = os.path.join(job_dir, "chunks") 
        padding_info_dir = os.path.join(job_dir, "padding_info")
        inference_output_dir = os.path.join(job_dir, "inference_output")
        final_output_path = os.path.join(job_dir, "final_output.tiff")
        
        for dir_path in [job_dir, chunks_dir, padding_info_dir, inference_output_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Initialize status
        self.update_status(
            status="running",
            progress=0,
            message="Starting processing pipeline",
            current_step="initialization",
            job_id=job_id,
            error=None
        )
        
        logger.info(f"Starting job {job_id} for file: {input_file_path}")
        
        # Start processing thread
        processing_thread = threading.Thread(
            target=self._run_processing_pipeline,
            args=(input_file_path, job_id, chunks_dir, padding_info_dir, 
                  inference_output_dir, final_output_path, chunk_size, num_workers)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        return job_id
        
    def _run_processing_pipeline(self, input_file_path, job_id, chunks_dir, 
                               padding_info_dir, inference_output_dir, 
                               final_output_path, chunk_size, num_workers):
        """
        Internal method to run the full processing pipeline.
        """

        job_dir = os.path.dirname(final_output_path)

        try:
            # Step 1: Preprocessing - Chunk Extraction
            self.update_status(
                progress=5,
                message="Starting image chunking",
                current_step="preprocessing"
            )
            
            logger.info(f"Job {job_id}: Starting preprocessing")
            
            # Dynamically import the preprocessing module
            try:
                # Try to import from project root first
                sys.path.insert(0, project_root)
                from chunk_extraction import parallel_chunk_extraction
            except ImportError:
                # Fall back to trying from current directory
                from chunk_extraction import parallel_chunk_extraction
            
            # Run preprocessing
            chunk_result = parallel_chunk_extraction(
                input_file_path,
                chunks_dir,
                padding_info_dir=padding_info_dir,
                chunk_size=chunk_size,
                num_workers=num_workers
            )
            
            self.update_status(
                progress=35,
                message=f"Chunking complete: {chunk_result['successful']}/{chunk_result['total_chunks']} chunks processed",
            )
            
            logger.info(f"Job {job_id}: Preprocessing complete - {chunk_result['successful']}/{chunk_result['total_chunks']} chunks")
            
            # Step 2: Run Inference
            self.update_status(
                progress=40,
                message="Starting inference on image chunks",
                current_step="inference"
            )
            
            logger.info(f"Job {job_id}: Starting inference with model {self.model_config['name']}")
            
            # Prepare inference command based on model type
            model_type = self.model_config.get("type", "diffusion")
            script_path = self.model_config.get("script_path")
            
            # Base command
            inference_cmd = [
                "torchrun",
                f"--nproc_per_node={self.gpu_count}",
                script_path
            ]
            
            # Add standard parameters
            inference_cmd.extend([
                f"--img_path={chunks_dir}/",
                f"--pretrained_path={self.model_config.get('checkpoint_path')}",
                f"--output_path={inference_output_dir}",
            ])
            
            # Add prompt if model type uses it
            if model_type == "diffusion":
                inference_cmd.append(f"--prompt={self.model_config.get('prompt', '')}")
            
            # Add any additional parameters from model config
            for param_name, param_value in self.model_config.get("params", {}).items():
                inference_cmd.append(f"--{param_name}={param_value}")
            
            logger.info(f"Job {job_id}: Running inference command: {' '.join(inference_cmd)}")
            
            # Run inference subprocess
            self._current_process = subprocess.Popen(
                inference_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor inference progress
            for line in self._current_process.stdout:
                if "%" in line:
                    try:
                        # Extract progress percentage if available
                        progress_str = line.split("%")[0].strip().split(" ")[-1]
                        progress_value = float(progress_str)
                        inference_progress = 40 + (progress_value * 0.3)  # Scale to our progress range
                        self.update_status(
                            progress=inference_progress,
                            message=f"Inference progress: {progress_value:.1f}%"
                        )
                    except (ValueError, IndexError):
                        pass
                
                # Log output for debugging
                logger.debug(f"Inference output: {line.strip()}")
            
            # Wait for process to complete
            self._current_process.wait()
            
            if self._current_process.returncode != 0:
                stderr = self._current_process.stderr.read()
                error_msg = f"Inference failed with error code {self._current_process.returncode}: {stderr}"
                logger.error(f"Job {job_id}: {error_msg}")
                raise Exception(error_msg)
            
            self._current_process = None
            
            self.update_status(
                progress=70,
                message="Inference complete",
            )
            
            logger.info(f"Job {job_id}: Inference complete")
            
            # Step 3: Restitching
            self.update_status(
                progress=75,
                message="Starting image restitching",
                current_step="restitching"
            )
            
            logger.info(f"Job {job_id}: Starting restitching")
            
            # Import the restitching module
            try:
                # Try to import from project root first
                sys.path.insert(0, project_root)
                from stitch_image import stitch_image
            except ImportError:
                # Fall back to trying from current directory
                from stitch_image import stitch_image
            
            # Run restitching
            stitch_result = stitch_image(
                inference_output_dir,
                padding_info_dir,
                final_output_path,
                tile_size=240
            )
            
            logger.info(f"Job {job_id}: Restitching complete - Output size: {stitch_result['width']}x{stitch_result['height']}")
            
            # Processing complete
            self.update_status(
                status="complete",
                progress=100,
                message=f"Processing complete. Output saved to {final_output_path}",
                current_step="complete"
            )
            
            # Create metadata file
            metadata = {
                "job_id": job_id,
                "input_file": os.path.basename(input_file_path),
                "output_file": os.path.basename(final_output_path),
                "model": self.model_config.get("name"),
                "model_id": self.model_config.get("id"),
                "chunking": chunk_result,
                "stitching": stitch_result,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success"
            }
            
            with open(os.path.join(job_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Job {job_id}: Processing complete")
                
        except Exception as e:
            # Handle errors
            error_message = str(e)
            logger.error(f"Job {job_id}: Processing failed - {error_message}", exc_info=True)
            
            self.update_status(
                status="error",
                message=f"Processing failed: {error_message}",
                current_step="error",
                error=error_message
            )
            
            # Create error log
            with open(os.path.join(job_dir, "error.log"), "w") as f:
                f.write(f"Error: {error_message}\n")
                
            return False
            
        return True
    
    def cancel_processing(self, job_id):
        """
        Attempt to cancel an in-progress job
        """
        logger.info(f"Attempting to cancel job {job_id}")
        
        if self.status["job_id"] == job_id and self.status["status"] == "running":
            # Kill the current subprocess if it exists
            if self._current_process is not None:
                try:
                    self._current_process.terminate()
                    self._current_process = None
                    logger.info(f"Job {job_id}: Process terminated")
                except Exception as e:
                    logger.error(f"Error terminating process: {str(e)}")
            
            self.update_status(
                status="cancelled",
                message="Processing was cancelled by user",
                current_step="cancelled"
            )
            
            logger.info(f"Job {job_id}: Marked as cancelled")
            return True
            
        logger.info(f"Job {job_id}: Cannot cancel (not running or different job)")
        return False
    
    def get_job_results(self, job_id):
        """
        Get the results for a completed job
        """
        job_dir = os.path.join(self.base_output_dir, job_id)
        metadata_path = os.path.join(job_dir, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        
        # If metadata doesn't exist, return the status
        if self.status["job_id"] == job_id:
            return self.get_status()
            
        return {"status": "not_found", "message": f"Job {job_id} not found"}