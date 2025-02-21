# model_definitions.py
"""
This file contains definitions for different models that can be used with the
WSI processing pipeline. Add new models here to make them available in the UI.
"""
import os
import json
import logging
import sys

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to store custom model definitions
CUSTOM_MODELS_FILE = os.environ.get('CUSTOM_MODELS_FILE', os.path.join(os.path.dirname(__file__), 'custom_models.json'))

# Base model configuration format:
# {
#     "id": "unique-model-id",
#     "name": "Human Readable Name",
#     "checkpoint_path": "./checkpoints/model.pkl",
#     "script_path": "inference_ddp.py",
#     "prompt": "Text prompt if applicable",
#     "type": "model type (diffusion, classifier, segmentation, etc.)",
#     "description": "Detailed description of what the model does",
#     "params": {"param1": "value1", "param2": "value2"}
# }

DEFAULT_MODELS = [
    {
        "id": "diffusion-ffpe",
        "name": "Diffusion FFPE",
        "checkpoint_path": os.path.join(project_root, "checkpoints/model.pkl"),
        "script_path": os.path.join(project_root, "inference_ddp.py"),
        "prompt": "paraffin section",
        "type": "diffusion",
        "description": "Default diffusion model for FFPE to H&E transformation",
        "params": {}
    }
]

def _load_custom_models():
    """Load custom models from JSON file if it exists"""
    if not os.path.exists(CUSTOM_MODELS_FILE):
        return []
    
    try:
        with open(CUSTOM_MODELS_FILE, 'r') as f:
            custom_models = json.load(f)
        logger.info(f"Loaded {len(custom_models)} custom models from {CUSTOM_MODELS_FILE}")
        return custom_models
    except Exception as e:
        logger.error(f"Error loading custom models: {str(e)}")
        return []

def _save_custom_models(custom_models):
    """Save custom models to JSON file"""
    try:
        with open(CUSTOM_MODELS_FILE, 'w') as f:
            json.dump(custom_models, f, indent=2)
        logger.info(f"Saved {len(custom_models)} custom models to {CUSTOM_MODELS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving custom models: {str(e)}")
        return False

def get_available_models():
    """Return the list of available models (default + custom)"""
    custom_models = _load_custom_models()
    # Use a dictionary to merge models, with custom models overriding defaults if they have the same ID
    models_dict = {model['id']: model for model in DEFAULT_MODELS}
    models_dict.update({model['id']: model for model in custom_models})
    return list(models_dict.values())

def get_model_by_id(model_id):
    """Get a specific model by ID"""
    for model in get_available_models():
        if model['id'] == model_id:
            return model
    return None

def add_custom_model(model_config):
    """
    Add a custom model configuration
    
    Args:
        model_config: Dictionary with model configuration
        
    Returns:
        bool: True if added successfully
    """
    model_id = model_config.get('id')
    if not model_id:
        raise ValueError("Model configuration must include an 'id' field")
    
    # Load existing custom models
    custom_models = _load_custom_models()
    
    # Check if model with this ID already exists in custom models
    for i, model in enumerate(custom_models):
        if model['id'] == model_id:
            # Replace existing model
            custom_models[i] = model_config
            logger.info(f"Updated existing custom model: {model_id}")
            return _save_custom_models(custom_models)
    
    # Add new model
    custom_models.append(model_config)
    logger.info(f"Added new custom model: {model_id}")
    return _save_custom_models(custom_models)

def remove_custom_model(model_id):
    """
    Remove a custom model by ID
    
    Args:
        model_id: ID of the model to remove
        
    Returns:
        bool: True if removed successfully, False if not found
    """
    # Load existing custom models
    custom_models = _load_custom_models()
    
    # Find and remove the model
    for i, model in enumerate(custom_models):
        if model['id'] == model_id:
            custom_models.pop(i)
            logger.info(f"Removed custom model: {model_id}")
            return _save_custom_models(custom_models)
    
    logger.warning(f"Custom model {model_id} not found for removal")
    return False

# Check for model checkpoints on startup
def validate_models():
    """Validate that model checkpoint files exist"""
    models = get_available_models()
    valid_models = []
    
    for model in models:
        checkpoint_path = model.get('checkpoint_path')
        if not checkpoint_path:
            logger.warning(f"Model {model['id']} has no checkpoint path")
            continue
            
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found for model {model['id']}: {checkpoint_path}")
            continue
            
        valid_models.append(model)
    
    logger.info(f"Validated {len(valid_models)} of {len(models)} models")
    return valid_models

# Startup validation of models
if __name__ == "__main__":
    validate_models()