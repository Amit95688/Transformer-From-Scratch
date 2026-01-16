"""Centralized logging for MLOps"""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logging(name: str) -> logging.Logger:
    """Setup structured logging"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # File handler with JSON formatting
    fh = logging.FileHandler(log_dir / "app.log")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger


def log_prediction(input_text: str, output_text: str, latency_ms: float, model_version: str) -> None:
    """Log prediction with structured format"""
    logger = logging.getLogger("predictions")
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_text[:100],  # Limit input length
        "output": output_text[:100],
        "latency_ms": latency_ms,
        "model_version": model_version
    }
    
    logger.info(json.dumps(log_entry))


def log_training_event(event: str, metrics: dict) -> None:
    """Log training events"""
    logger = logging.getLogger("training")
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "metrics": metrics
    }
    
    logger.info(json.dumps(log_entry))


if __name__ == '__main__':
    # Test logging functionality
    logger = setup_logging("test")
    logger.info("Logger test successful")
    
    log_prediction("Hello", "नमस्ते", 45.5, "v1.0.0")
    log_training_event("epoch_complete", {"epoch": 1, "loss": 0.5})
    print("Logging utilities tested successfully!")
