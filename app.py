"""
Main Flask Application for Transformer Model
Entry point for the web application
"""
from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from TransformerModel.pipelines.prediction_pipeline import PredictPipeline, CustomData
from TransformerModel.logger import logging
from TransformerModel.exception import CustomException

app = Flask(__name__)


@app.route('/')
def home():
    """
    Home page route
    """
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Prediction endpoint
    """
    try:
        if request.method == 'GET':
            return render_template('translate.html')
        else:
            # Get input from form
            source_text = request.form.get('source_text')
            
            if not source_text:
                return jsonify({'error': 'No input text provided'}), 400
            
            # Create custom data object
            data = CustomData(source_text=source_text)
            
            # Initialize prediction pipeline
            predict_pipeline = PredictPipeline(
                model_path='artifacts/model.pth',
                tokenizer_src_path='tokenizer_en.json',
                tokenizer_tgt_path='tokenizer_hi.json'
            )
            
            # Get prediction
            result = predict_pipeline.predict(data.get_data_as_input())
            
            logging.info(f"Translation completed: {source_text} -> {result}")
            
            return render_template('translate.html', 
                                 source_text=source_text, 
                                 translation=result)
            
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise CustomException(e, sys)


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({'status': 'healthy', 'service': 'transformer-api'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
