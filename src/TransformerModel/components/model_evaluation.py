"""
Model Evaluation Component for Transformer Model
"""
import os
import sys
import torch
from pathlib import Path
from dataclasses import dataclass

from TransformerModel.logger import logging
from TransformerModel.exception import CustomException
from TransformerModel.components.model_trainer import build_transformer
from TransformerModel.components.data_transformation import casual_mask


@dataclass
class ModelEvaluationConfig:
    """
    Configuration class for Model Evaluation
    """
    model_path: str = os.path.join('artifacts', 'model.pth')
    metrics_path: str = os.path.join('artifacts', 'metrics.txt')


class ModelEvaluation:
    """
    This class handles Model Evaluation
    """
    
    def __init__(self):
        self.evaluation_config = ModelEvaluationConfig()
    
    def calculate_bleu_score(self, predictions, references):
        """
        Calculate BLEU score for translation quality
        """
        try:
            # BLEU score calculation logic
            logging.info("BLEU score calculated")
            return 0.0  # Placeholder
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_evaluation(self, model, test_dataloader, tokenizer_tgt, device):
        """
        Evaluate the model on test data
        
        Args:
            model: Trained transformer model
            test_dataloader: DataLoader for test data
            tokenizer_tgt: Target language tokenizer
            device: torch device
        """
        logging.info("Model evaluation started")
        
        try:
            model.eval()
            total_loss = 0
            count = 0
            
            with torch.no_grad():
                for batch in test_dataloader:
                    encoder_input = batch['encoder_input'].to(device)
                    decoder_input = batch['decoder_input'].to(device)
                    encoder_mask = batch['enc_mask'].to(device)
                    decoder_mask = batch['dec_mask'].to(device)
                    label = batch['label'].to(device)
                    
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
                    proj_output = model.project(decoder_output)
                    
                    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'))
                    loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                    
                    total_loss += loss.item()
                    count += 1
            
            avg_loss = total_loss / count if count > 0 else 0
            
            logging.info(f"Average test loss: {avg_loss}")
            
            # Save metrics
            with open(self.evaluation_config.metrics_path, 'w') as f:
                f.write(f"Average Loss: {avg_loss}\n")
            
            logging.info("Model evaluation completed")
            
        except Exception as e:
            raise CustomException(e, sys)
