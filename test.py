"""
Test script for Transformer Model
"""
from TransformerModel.pipelines.training_pipeline import TrainingPipeline
from TransformerModel.pipelines.prediction_pipeline import CustomData, PredictPipeline
from TransformerModel.logger import logging


def test_training_pipeline():
    """Test the training pipeline"""
    print("=" * 50)
    print("Testing Training Pipeline")
    print("=" * 50)
    
    training_pipeline = TrainingPipeline()
    # training_pipeline.start()  # Uncomment to run full training
    print("Training pipeline initialized successfully!")


def test_prediction_pipeline():
    """Test the prediction pipeline"""
    print("\n" + "=" * 50)
    print("Testing Prediction Pipeline")
    print("=" * 50)
    
    try:
        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline(
            model_path='artifacts/model.pth',
            tokenizer_src_path='tokenizer_en.json',
            tokenizer_tgt_path='tokenizer_hi.json'
        )
        
        # Create sample input
        data = CustomData(source_text="Hello, how are you?")
        input_text = data.get_data_as_input()
        
        print(f"\nInput Text: {input_text}")
        
        # Get prediction
        translation = predict_pipeline.predict(input_text)
        
        print(f"Translation: {translation}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print("Make sure you have trained the model and tokenizers are available.")


if __name__ == '__main__':
    print("Starting Transformer Model Tests...")
    print("=" * 50)
    
    # Test training pipeline
    test_training_pipeline()
    
    # Test prediction pipeline
    test_prediction_pipeline()
    
    print("\n" + "=" * 50)
    print("Testing Complete")
    print("=" * 50)
