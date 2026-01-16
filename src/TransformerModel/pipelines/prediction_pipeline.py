"""
Prediction Pipeline for Transformer Model
"""
import sys
from pathlib import Path
import torch
from dataclasses import dataclass

# Add project root and src directory to sys.path for imports
_here = Path(__file__).resolve()
_project_root = _here.parents[3]
_src_dir = _here.parents[2]
for p in (str(_project_root), str(_src_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from TransformerModel.components.model_trainer import build_transformer
from TransformerModel.components.data_transformation import casual_mask
from TransformerModel.logger import logging
from TransformerModel.exception import CustomException
from tokenizers import Tokenizer


@dataclass
class CustomData:
    """
    Custom data class for prediction input
    """
    source_text: str
    
    def get_data_as_input(self):
        return self.source_text


class PredictPipeline:
    """
    Pipeline for making predictions with trained Transformer model
    """
    
    def __init__(self, model_path: str, tokenizer_src_path: str, tokenizer_tgt_path: str, device: str = 'cpu'):
        try:
            self.device = torch.device(device)
            self.tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
            self.tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)
            
            # Load model
            self.model = self.load_model(model_path)
            self.model.eval()
            logging.info("Prediction pipeline initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_model(self, model_path: str):
        try:
            # Model loading logic here
            model = build_transformer(
                self.tokenizer_src.get_vocab_size(),
                self.tokenizer_tgt.get_vocab_size(),
                512,  # seq_len
                512,  # seq_len
                512   # d_model
            )
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            logging.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            raise CustomException(e, sys)
    
    def greedy_decode(self, encoder_input, encoder_mask, max_len=512):
        try:
            sos_idx = self.tokenizer_tgt.token_to_id("[SOS]")
            eos_idx = self.tokenizer_tgt.token_to_id("[EOS]")
            
            encoder_output = self.model.encode(encoder_input, encoder_mask)
            decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(self.device)
            
            while True:
                if decoder_input.size(1) >= max_len:
                    break
                
                decoder_mask = casual_mask(decoder_input.size(1)).type_as(encoder_input).to(self.device)
                decoder_output = self.model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
                projected_output = self.model.project(decoder_output)
                
                _, next_word = torch.max(projected_output[:, -1, :], dim=1)
                next_word = next_word.item()
                
                decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word).to(self.device)], dim=1)
                
                if next_word == eos_idx:
                    break
            
            return decoder_input.squeeze(0)
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, source_text: str):
        try:
            # Tokenize input
            encoder_input = self.tokenizer_src.encode(source_text)
            encoder_input = torch.tensor(encoder_input.ids).unsqueeze(0).to(self.device)
            
            # Create mask
            encoder_mask = (encoder_input != self.tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int().to(self.device)
            
            # Generate translation
            with torch.no_grad():
                output = self.greedy_decode(encoder_input, encoder_mask)
            
            # Decode output
            translation = self.tokenizer_tgt.decode(output.detach().cpu().numpy())
            logging.info(f"Translation generated for input: {source_text}")
            
            return translation
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    # Test prediction pipeline
    print("Prediction Pipeline component ready")
    print("To use: Initialize with model_path, tokenizer_src_path, tokenizer_tgt_path")
    print("Example:")
    print("  pipeline = PredictPipeline('model.pth', 'tokenizer_en.json', 'tokenizer_hi.json')")
    print("  translation = pipeline.predict('Hello, how are you?')")
