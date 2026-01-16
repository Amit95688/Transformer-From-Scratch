"""
Training Pipeline for Transformer Model
"""
import sys
from pathlib import Path

# Add project root and src directory to sys.path for imports
_here = Path(__file__).resolve()
_project_root = _here.parents[3]  # /home/.../transformer
_src_dir = _here.parents[2]       # /home/.../transformer/src
for p in (str(_project_root), str(_src_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from TransformerModel.components.data_transformation import BilingualDataset, casual_mask
from TransformerModel.components.model_trainer import build_transformer
from TransformerModel.logger import logging
from TransformerModel.exception import CustomException
from config.config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


class TrainingPipeline:
    """
    Main training pipeline for the Transformer model
    """
    
    def __init__(self):
        self.config = get_config()
        logging.info("Training pipeline initialized")
    
    def get_all_sentence(self, ds, lang):
        for item in ds:
            yield item['translation'][lang]
    
    def get_or_build_tokenizer(self, config, ds, lang):
        try:
            tokenizer_path = Path(config['tokenizer_file'].format(lang=lang))
            if not Path.exists(tokenizer_path):
                tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
                tokenizer.pre_tokenizer = Whitespace()
                trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
                tokenizer.train_from_iterator(self.get_all_sentence(ds, lang), trainer=trainer)
                tokenizer.save(str(tokenizer_path))
                logging.info(f"Tokenizer created and saved for {lang}")
            else:
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                logging.info(f"Tokenizer loaded for {lang}")
            return tokenizer
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_ds(self, config):
        try:
            # Load dataset - try different dataset sources
            try:
                ds_raw = load_dataset(config['datasource'], f"{config['lang_src']}-{config['lang_tgt']}", split='train')
            except (ValueError, AttributeError):
                # If config name not found, use 'default'
                ds_raw = load_dataset(config['datasource'], 'default', split='train')
                # Filter for English-Hindi if needed
                ds_raw = ds_raw.filter(lambda x: x.get('src', '').startswith('en') and x.get('tgt', '').startswith('hi') if 'src' in x else True)
            
            tokenizer_src = self.get_or_build_tokenizer(config, ds_raw, config['lang_src'])
            tokenizer_tgt = self.get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
            
            train_ds_size = int(0.9 * len(ds_raw))
            val_ds_size = len(ds_raw) - train_ds_size
            train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
            
            train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, 
                                       config['lang_src'], config['lang_tgt'], config['seq_len'])
            val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, 
                                     config['lang_src'], config['lang_tgt'], config['seq_len'])
            
            train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
            val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
            
            logging.info("Datasets and dataloaders created successfully")
            return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_validation(self, model, validation_df, tokenizer_src, tokenizer_tgt, max_len, device, writer, global_step, num_examples=2):
        """Run validation on a subset of validation data"""
        model.eval()
        count = 0
        
        with torch.no_grad():
            for batch in validation_df:
                count += 1
                
                encoder_input = batch['encoder_input'].to(device)
                encoder_mask = batch['enc_mask'].to(device)
                
                # Greedy decode
                sos_idx = tokenizer_tgt.token_to_id("[SOS]")
                eos_idx = tokenizer_tgt.token_to_id("[EOS]")
                
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
                
                while True:
                    if decoder_input.size(1) >= max_len:
                        break
                    
                    decoder_mask = casual_mask(decoder_input.size(1)).type_as(encoder_input).to(device)
                    decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
                    projected_output = model.project(decoder_output)
                    
                    _, next_word = torch.max(projected_output[:, -1, :], dim=1)
                    next_word = next_word.item()
                    
                    decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word).to(device)], dim=1)
                    
                    if next_word == eos_idx:
                        break
                
                source_text = batch['src_text'][0]
                target_text = batch['tgt_text'][0]
                model_out_text = tokenizer_tgt.decode(decoder_input.squeeze(0).detach().cpu().numpy())
                
                logging.info('-' * 80)
                logging.info(f"SOURCE: {source_text}")
                logging.info(f"TARGET: {target_text}")
                logging.info(f"PREDICTED: {model_out_text}")
                logging.info('-' * 80)
                
                if count == num_examples:
                    break
    
    def start(self):
        """
        Start the training pipeline
        """
        try:
            logging.info("Training started")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f"Using device: {device}")
            
            config = self.config
            Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
            
            train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = self.get_ds(config)
            
            model = build_transformer(
                tokenizer_src.get_vocab_size(),
                tokenizer_tgt.get_vocab_size(),
                config['seq_len'],
                config['seq_len'],
                config['d_model']
            ).to(device)
            
            writer = SummaryWriter(config['experiment_name'])
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1)
            
            initial_epoch = 0
            global_step = 0
            
            # Load checkpoint if exists
            if config.get('preload'):
                model_filename = get_weights_file_path(config, config['preload'])
                logging.info(f"Loading model weights from {model_filename}")
                state = torch.load(model_filename, map_location=device)
                model.load_state_dict(state['model_state_dict'])
                optimizer.load_state_dict(state['optimizer_state_dict'])
                initial_epoch = state['epoch']
                global_step = state['global_step']
            
            # Training loop
            for epoch in range(initial_epoch, config['num_epochs']):
                model.train()
                batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
                
                for batch in batch_iterator:
                    encoder_input = batch['encoder_input'].to(device)
                    decoder_input = batch['decoder_input'].to(device)
                    encoder_mask = batch['enc_mask'].to(device)
                    decoder_mask = batch['dec_mask'].to(device)
                    label = batch['label'].to(device)
                    
                    # Forward pass
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
                    projected_output = model.project(decoder_output)
                    
                    # Calculate loss
                    loss = loss_fn(projected_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                    
                    # Check for NaN
                    if torch.isnan(loss):
                        logging.warning("NaN loss detected! Skipping batch...")
                        continue
                    
                    batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                    writer.add_scalar('Train/Loss', loss.item(), global_step)
                    writer.flush()
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                # Run validation
                self.run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                                  config['seq_len'], device, writer, global_step)
                
                # Save checkpoint
                model_filename = get_weights_file_path(config, f"{epoch+1:02d}")
                logging.info(f"Saving model checkpoint to {model_filename}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                }, model_filename)
            
            logging.info("Training completed successfully")
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    
    training_pipeline = TrainingPipeline()
    training_pipeline.start()