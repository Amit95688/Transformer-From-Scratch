# MLflow Integration Guide

## What is MLflow?
MLflow is an open platform for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment.

## Features Integrated

### 1. **Experiment Tracking**
- All training runs are logged under the experiment name from config
- Located in `mlruns/` directory (local file-based backend)

### 2. **Metrics Logging**
- `train_loss`: Training loss for each batch step
- `epoch`: Epoch number at the end of each epoch

### 3. **Parameters Logging**
- All configuration parameters (batch_size, learning_rate, seq_length, etc.)
- Useful for comparing different hyperparameter settings

### 4. **Artifacts**
- Model checkpoints saved automatically
- Latest model logged in PyTorch format

## Quick Start

### 1. Install MLflow
```bash
pip install -r requirements.txt  # or requirements_dev.txt
```

### 2. Run Training
```bash
python train.py
```

### 3. View MLflow UI
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser

## MLflow UI Features

### Experiments Tab
- View all experiments
- Compare metrics across runs
- See parameters used in each run

### Runs Tab
- Detailed metrics charts
- Download logged artifacts
- View configuration used

## Docker Usage

```bash
# Build image
docker build -t transformer:latest .

# Run training with MLflow logging
docker run --rm -v $(pwd)/mlruns:/app/mlruns transformer:latest

# View results on host machine
mlflow ui
```

## Advanced Usage

### Log Custom Metrics
```python
import mlflow

# Log a custom metric
mlflow.log_metric('custom_metric_name', value, step=global_step)

# Log multiple metrics at once
mlflow.log_metrics({'metric1': val1, 'metric2': val2}, step=step)
```

### Add Tags to Runs
```python
mlflow.set_tag('model_type', 'transformer')
mlflow.set_tag('language_pair', 'en-hi')
```

### Log Additional Artifacts
```python
# Log a file
mlflow.log_artifact('path/to/file.txt')

# Log a directory
mlflow.log_artifacts('path/to/directory/')
```

## File Structure

```
.
├── mlruns/                      # MLflow tracking directory
│   └── [experiment_id]/
│       └── [run_id]/
│           ├── artifacts/       # Stored artifacts
│           ├── metrics/         # Logged metrics
│           ├── params/          # Logged parameters
│           └── meta.yaml        # Run metadata
├── models/
├── runs/                        # TensorBoard logs
└── train.py
```

## Comparing Runs

1. Open MLflow UI: `mlflow ui`
2. Go to the experiment
3. Select multiple runs using checkboxes
4. Click "Compare" to see side-by-side comparison

## Model Registry (Optional)

For production deployment, register models:

```python
import mlflow

# Register model
mlflow.register_model('runs:/run_id/model_latest', 'transformer_en_hi')

# Later, load it
loaded_model = mlflow.pytorch.load_model('models:/transformer_en_hi/production')
```

## Troubleshooting

### MLflow UI not loading
```bash
# Check if port 5000 is in use
lsof -i :5000
# Kill process and restart
mlflow ui
```

### Artifacts not logging
- Ensure `mlruns/` directory exists and is writable
- Check file paths are absolute or relative to working directory

### Run with different tracking URI
```bash
mlflow.set_tracking_uri('http://localhost:5000')  # Remote server
# or
mlflow.set_tracking_uri('file:/path/to/mlruns')   # Different directory
```
