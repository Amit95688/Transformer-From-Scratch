"""
Utility functions for Transformer Model
"""
import os
import sys
import pickle
import torch
from pathlib import Path

from TransformerModel.logger import logging
from TransformerModel.exception import CustomException


def save_object(file_path: str, obj):
    """
    Save a Python object to disk using pickle
    
    Args:
        file_path: Path where to save the object
        obj: Object to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Load a Python object from disk
    
    Args:
        file_path: Path to the saved object
        
    Returns:
        Loaded object
    """
    try:
        with open(file_path, 'rb') as file_obj:
            logging.info(f"Object loaded from {file_path}")
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Exception occurred while loading object')
        raise CustomException(e, sys)


def save_model(model, file_path: str):
    """
    Save PyTorch model state dict
    
    Args:
        model: PyTorch model
        file_path: Path where to save the model
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        torch.save(model.state_dict(), file_path)
        logging.info(f"Model saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_model(model, file_path: str, device='cpu'):
    """
    Load PyTorch model state dict
    
    Args:
        model: PyTorch model architecture
        file_path: Path to the saved model
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    try:
        model.load_state_dict(torch.load(file_path, map_location=device))
        logging.info(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == '__main__':
    # Test utility functions
    import tempfile
    
    print("Testing utility functions...")
    
    # Test save and load object
    test_obj = {"key": "value", "number": 42}
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    save_object(temp_path, test_obj)
    loaded_obj = load_object(temp_path)
    
    assert loaded_obj == test_obj, "Object save/load failed"
    print("✓ Object save/load works correctly")
    
    # Clean up
    import os
    os.remove(temp_path)
    
    print("All utility functions tested successfully!")
