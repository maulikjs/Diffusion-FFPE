# Create a simple API wrapper to integrate with your UI

import os
import json
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from processing_controller import ProcessingController

app = Flask(__name__)

# Define available models
AVAILABLE_MODELS = [
    {
        "id": "diffusion-ffpe",
        "name": "Diffusion FFPE",
        "checkpoint_path": "./checkpoints/model.pkl",
        "script_path": "inference_ddp.py",
        "prompt": "paraffin section",
        "type": "diffusion",
        "description": "Default FFPE diffusion model",
        "params": {}
    },
    # Example of another model with different configuration
    {
        "id": "diffusion-he",
        "name": "Diffusion H&E",
        "checkpoint_path": "./checkpoints/he_model.pkl",
        "script_path": "inference_ddp.py",
        "prompt": "hematoxylin and eosin stain",
        "type": "diffusion",
        "description": "H&E stain generation model",
        "params": {}
    }
    # Add more model configurations as needed
]

# Initialize the processing controller with default model
controller = ProcessingController(
    model_config=AVAILABLE_MODELS[0],
    output_dir="./results"
)

# Configure upload settings
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'svs', 'tiff', 'tif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB max upload

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
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
        
        # Start processing the file
        job_id = controller.process_image(
            file_path,
            chunk_size=int(request.form.get('chunk_size', 512)),
            num_workers=int(request.form.get('num_workers', 4))
        )
        
        return jsonify({
            'job_id': job_id,
            'message': 'Processing started',
            'status': controller.get_status()
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get the current status of a processing job"""
    if controller.get_status().get('job_id') == job_id:
        return jsonify(controller.get_status())
    
    # Check if the job has completed
    results = controller.get_job_results(job_id)
    return jsonify(results)

@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a running job"""
    cancelled = controller.cancel_processing(job_id)
    if cancelled:
        return jsonify({'message': 'Job cancelled', 'status': controller.get_status()})
    return jsonify({'error': 'Cannot cancel job', 'status': controller.get_status()}), 400

@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """Get the final result image for a completed job"""
    # Check if job exists and is complete
    job_results = controller.get_job_results(job_id)
    
    if job_results.get('status') != 'success':
        return jsonify({'error': 'Result not available', 'status': job_results}), 404
    
    # Get path to the final output image
    output_path = os.path.join(controller.base_output_dir, job_id, "final_output.tiff")
    
    if not os.path.exists(output_path):
        return jsonify({'error': 'Output file not found'}), 404
        
    return send_file(output_path, as_attachment=True)

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get a list of all available models"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'current_model': controller.get_model_config()
    })

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model_details(model_id):
    """Get details for a specific model"""
    model = next((m for m in AVAILABLE_MODELS if m['id'] == model_id), None)
    if not model:
        return jsonify({'error': f'Model {model_id} not found'}), 404
    
    return jsonify(model)

@app.route('/api/models/select/<model_id>', methods=['POST'])
def select_model(model_id):
    """Select a model to use for processing"""
    model = next((m for m in AVAILABLE_MODELS if m['id'] == model_id), None)
    if not model:
        return jsonify({'error': f'Model {model_id} not found'}), 404
    
    # Update the controller with the selected model config
    updated_config = controller.set_model_config(model)
    
    return jsonify({
        'message': f'Model {model["name"]} selected',
        'model': updated_config
    })

@app.route('/api/configure', methods=['POST'])
def configure_processor():
    """Update the processor configuration"""
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
    
    # Return the current configuration
    return jsonify({
        'message': 'Configuration updated',
        'model_config': controller.get_model_config(),
        'gpu_count': controller.gpu_count
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)