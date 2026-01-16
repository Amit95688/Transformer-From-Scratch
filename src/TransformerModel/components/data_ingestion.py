"""
Data Ingestion Component for Transformer Model
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass

from datasets import load_dataset
from TransformerModel.logger import logging
from TransformerModel.exception import CustomException


@dataclass
class DataIngestionConfig:
    """
    Configuration class for Data Ingestion
    """
    raw_data_path: str = os.path.join('artifacts', 'raw_data')
    train_data_path: str = os.path.join('artifacts', 'train_data')
    test_data_path: str = os.path.join('artifacts', 'test_data')


class DataIngestion:
    """
    This class handles Data Ingestion for translation datasets
    """
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self, datasource: str, lang_pair: str):
        """
        Downloads and prepares the dataset
        
        Args:
            datasource: Name of the dataset (e.g., 'opus_books')
            lang_pair: Language pair (e.g., 'en-hi')
        
        Returns:
            tuple: Paths to train and test datasets
        """
        logging.info("Data ingestion started")
        
        try:
            # Load dataset
            ds_raw = load_dataset(datasource, lang_pair, split='train')
            logging.info(f"Dataset {datasource} loaded successfully")
            
            # Create directories
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Split dataset
            train_test_split = ds_raw.train_test_split(test_size=0.1, seed=42)
            train_ds = train_test_split['train']
            test_ds = train_test_split['test']
            
            logging.info(f"Dataset split: {len(train_ds)} train, {len(test_ds)} test")
            
            # Save datasets
            train_ds.save_to_disk(self.ingestion_config.train_data_path)
            test_ds.save_to_disk(self.ingestion_config.test_data_path)
            
            logging.info("Data ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    # Test data ingestion
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion(
        datasource='cfilt/iitb-english-hindi',
        lang_pair='translation'
    )
    print(f"Train data path: {train_path}")
    print(f"Test data path: {test_path}")
