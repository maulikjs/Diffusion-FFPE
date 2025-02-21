"""
API server for the WSI Processing Pipeline
"""
import os
import json
import time
import sys
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Add deployment dir to path
deployment_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(deployment_dir)

# Import processing controller and model definitions
from processing_controller import ProcessingController
from model_definitions import get_available_models, get_model_by_id, add_custom_model, validate_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get environment variables or use defaults
UPLOAD_DIR = os.environ.get('UPLOAD_DIR', os.path.join(project_root, 'data/uploads'))
RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join(project_root, 'data/results'))
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', os.path.join(project_root, 'checkpoints'))
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 4))
GPU_COUNT = int(os.environ.get('GPU_COUNT', 1))

# Create directories if they don't exist
for directory in [UPLOAD_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure upload settings
ALLOWED_EXTENSIONS = {'svs', 'tiff', 'tif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB max upload

# Validate models and get the list
AVAILABLE_MODELS = validate_models()
if not AVAILABLE_MODELS:
    logger.warning("No valid models found! Using default model configuration.")
    AVAILABLE_MODELS = get_available_models()

# Initialize the processing controller with the first model
controller = ProcessingController(
    model_config=AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None,
    output_dir=RESULTS_DIR
)

# Update GPU count from environment
controller.gpu_count = GPU_COUNT

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'models_available': len(AVAILABLE_MODELS),
        'current_model': controller.get_model_config().get('id'),
        'controller_status': controller.get_status().get('status')
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process a file"""
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            logger.info(f"File uploaded: {filename}")
            
            # Get processing parameters
            chunk_size = int(request.form.get('chunk_size', 512))
            num_workers = int(request.form.get('num_workers', MAX_WORKERS))
            model_id = request.form.get('model_id', None)
            
            # If model_id provided, switch to that model
            if model_id:
                model = get_model_by_id(model_id)
                if model:
                    controller.set_model_config(model)
                    logger.info(f"Using model: {model['name']}")
                else:
                    logger.warning(f"Model ID {model_id} not found, using current model")
            
            # Start processing the file
            job_id = controller.process_image(
                file_path,
                chunk_size=chunk_size,
                num_workers=num_workers
            )
            
            logger.info(f"Processing started with job ID: {job_id}")
            
            return jsonify({
                'job_id': job_id,
                'message': 'Processing started',
                'status': controller.get_status()
            })
        
        return jsonify({'error': 'File type not allowed'}), 400
    
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get the current status of a processing job"""
    try:
        if controller.get_status().get('job_id') == job_id:
            return jsonify(controller.get_status())
        
        # Check if the job has completed
        results = controller.get_job_results(job_id)
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in get_status: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a running job"""
    try:
        cancelled = controller.cancel_processing(job_id)
        if cancelled:
            logger.info(f"Job cancelled: {job_id}")
            return jsonify({'message': 'Job cancelled', 'status': controller.get_status()})
        logger.warning(f"Cannot cancel job: {job_id}")
        return jsonify({'error': 'Cannot cancel job', 'status': controller.get_status()}), 400
    
    except Exception as e:
        logger.error(f"Error in cancel_job: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """Get the final result image for a completed job"""
    try:
        # Check if job exists and is complete
        job_results = controller.get_job_results(job_id)
        
        if job_results.get('status') != 'success':
            return jsonify({'error': 'Result not available', 'status': job_results}), 404
        
        # Get path to the final output image
        output_path = os.path.join(controller.base_output_dir, job_id, "final_output.tiff")
        
        if not os.path.exists(output_path):
            return jsonify({'error': 'Output file not found'}), 404
            
        return send_file(output_path, as_attachment=True)
    
    except Exception as e:
        logger.error(f"Error in get_result: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    try:
        # Get all job directories
        if not os.path.exists(RESULTS_DIR):
            return jsonify({'jobs': []})
            
        jobs = []
        for job_id in os.listdir(RESULTS_DIR):
            job_dir = os.path.join(RESULTS_DIR, job_id)
            if not os.path.isdir(job_dir):
                continue
                
            # Check if metadata exists
            metadata_path = os.path.join(job_dir, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    jobs.append(metadata)
                except Exception as e:
                    logger.error(f"Error reading metadata for job {job_id}: {str(e)}")
            else:
                # Job might be in progress or failed
                jobs.append({
                    'job_id': job_id,
                    'status': 'unknown',
                    'message': 'Metadata not available'
                })
        
        return jsonify({'jobs': jobs})
    
    except Exception as e:
        logger.error(f"Error in list_jobs: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_available_models_api():
    """Get a list of all available models"""
    try:
        return jsonify({
            'models': get_available_models(),
            'current_model': controller.get_model_config()
        })
    
    except Exception as e:
        logger.error(f"Error in get_available_models_api: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model_details_api(model_id):
    """Get details for a specific model"""
    try:
        model = get_model_by_id(model_id)
        if not model:
            return jsonify({'error': f'Model {model_id} not found'}), 404
        
        return jsonify(model)
    
    except Exception as e:
        logger.error(f"Error in get_model_details_api: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/select/<model_id>', methods=['POST'])
def select_model_api(model_id):
    """Select a model to use for processing"""
    try:
        model = get_model_by_id(model_id)
        if not model:
            return jsonify({'error': f'Model {model_id} not found'}), 404
        
        # Update the controller with the selected model config
        updated_config = controller.set_model_config(model)
        logger.info(f"Model selected: {model['name']}")
        
        return jsonify({
            'message': f'Model {model["name"]} selected',
            'model': updated_config
        })
    
    except Exception as e:
        logger.error(f"Error in select_model_api: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/add', methods=['POST'])
def add_custom_model_api():
    """Add a custom model configuration"""
    try:
        model_config = request.json
        if not model_config:
            return jsonify({'error': 'No model configuration provided'}), 400
        
        # Validate required fields
        required_fields = ['id', 'name', 'checkpoint_path', 'script_path', 'type']
        for field in required_fields:
            if field not in model_config:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Add the model
        success = add_custom_model(model_config)
        if success:
            logger.info(f"Custom model added: {model_config['name']}")
            return jsonify({
                'message': 'Custom model added successfully',
                'model': model_config
            })
        
        return jsonify({'error': 'Failed to add custom model'}), 500
    
    except Exception as e:
        logger.error(f"Error in add_custom_model_api: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/configure', methods=['POST'])
def configure_processor_api():
    """Update the processor configuration"""
    try:
        data = request.json
        
        # Update GPU count if provided
        if 'gpu_count' in data:
            controller.gpu_count = int(data['gpu_count'])
        
        # Handle direct model configuration updates
        if 'model_config' in data:
            model_config = data['model_config']
            controller.set_model_config(model_config)
        
        # Handle individual model parameter updates
        elif 'model_params' in data:
            model_config = controller.get_model_config()
            for key, value in data['model_params'].items():
                if key == 'checkpoint_path':
                    model_config['checkpoint_path'] = value
                elif key == 'prompt':
                    model_config['prompt'] = value
                elif key == 'params':
                    # Update or add parameters
                    model_config['params'].update(value)
            
            controller.set_model_config(model_config)
        
        logger.info("Configuration updated")
        
        # Return the current configuration
        return jsonify({
            'message': 'Configuration updated',
            'model_config': controller.get_model_config(),
            'gpu_count': controller.gpu_count
        })
    
    except Exception as e:
        logger.error(f"Error in configure_processor_api: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Log startup information
    logger.info(f"Starting WSI Processing API on port {port}")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")
    logger.info(f"Max workers: {MAX_WORKERS}")
    logger.info(f"Available models: {len(AVAILABLE_MODELS)}")
    
    # Run the app
    app.run(debug=False, host='0.0.0.0', port=port)