# 🤖 Transformer Model - English to Hindi Translation

> Production-ready transformer model for bilingual machine translation with complete MLOps infrastructure

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen.svg)](#cicd)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [MLOps Infrastructure](#mlops-infrastructure)
- [Model Architecture](#model-architecture)
- [Testing](#testing)
- [Docker](#docker)
- [Contributing](#contributing)
- [Resources](#resources)

---

## 🎯 Overview

This project implements a **transformer-based machine translation model** (English ↔ Hindi) with a complete production-ready MLOps infrastructure. It demonstrates modern machine learning best practices including:

- ✅ Modular code architecture
- ✅ Comprehensive testing framework
- ✅ Automated CI/CD pipelines
- ✅ Experiment tracking & monitoring
- ✅ Professional documentation
- ✅ Docker containerization
- ✅ Data versioning with DVC
- ✅ Workflow orchestration with Airflow

---

## ✨ Features

### 🧠 Model Components
- **Transformer Architecture** - Multi-head attention, positional encoding
- **Bilingual Dataset** - English-Hindi parallel corpus
- **Tokenization** - SentencePiece tokenizers for both languages
- **Training Pipeline** - Distributed training with PyTorch
- **Evaluation Metrics** - BLEU score, perplexity tracking

### 🔄 MLOps Features
| Feature | Tool | Status |
|---------|------|--------|
| **Experiment Tracking** | MLflow | ✅ Implemented |
| **Data Versioning** | DVC | ✅ Configured |
| **Workflow Orchestration** | Airflow | ✅ DAGs Ready |
| **CI/CD Pipeline** | GitHub Actions | ✅ Active |
| **Monitoring & Logging** | Custom | ✅ Built-in |
| **Testing** | Pytest | ✅ Comprehensive |
| **Containerization** | Docker | ✅ Ready |

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/Amit95688/Transformer-From-Scratch.git
cd Transformer-From-Scratch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python scripts/train.py
```

### 3️⃣ Start Web Application

```bash
python main.py
# Visit http://localhost:5000
```

### 4️⃣ Run Tests

```bash
pip install -r requirements_dev.txt
pytest tests/ -v
```

---

## 📁 Project Structure

```
transformer/
├── src/                          # Source code
│   ├── core/                    # ML core components
│   │   ├── model.py             # Transformer architecture
│   │   ├── dataset.py           # Data loading & preprocessing
│   │   └── __init__.py
│   ├── web/                     # Flask web application
│   │   ├── app.py               # Main web app
│   │   └── __init__.py
│   ├── monitoring/              # MLOps monitoring
│   │   ├── logger.py            # Structured logging
│   │   ├── metrics.py           # Metrics collection
│   │   └── __init__.py
│   └── utils/                   # Helper utilities
│
├── config/                       # Configuration management
│   ├── config.py                # Main config file
│   └── __init__.py
│
├── data/                         # Data directory
│   └── tokenizers/              # Language tokenizers
│
├── scripts/                      # Standalone scripts
│   └── train.py                 # Training script
│
├── tests/                        # Test suite
│   ├── conftest.py              # Pytest fixtures
│   ├── test_model.py            # Model tests
│   ├── test_data.py             # Data tests
│   ├── test_monitoring.py       # Monitoring tests
│   └── test_model_artifacts.py  # Artifact tests
│
├── dags/                         # Airflow DAGs
│   └── training_pipeline_dag.py # Training orchestration
│
├── templates/                    # Flask HTML templates
│   ├── base.html
│   └── index.html
│
├── .github/workflows/            # CI/CD pipelines
│   ├── ci-cd.yml
│   └── model-validation.yml
│
├── main.py                       # Application entry point
├── requirements.txt              # Production dependencies
├── requirements_dev.txt          # Development dependencies
├── dvc.yaml                      # DVC pipeline
├── airflow.cfg                   # Airflow configuration
├── Dockerfile                    # Container definition
└── README.md                     # This file
```

---

## 💻 Installation

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Setup

```bash
# Clone repository
git clone https://github.com/Amit95688/Transformer-From-Scratch.git
cd Transformer-From-Scratch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.core.model import build_transformer; print('✓ Setup successful')"
```

---

## 🎮 Usage

### Training

```bash
python scripts/train.py
```

### Web Interface

```bash
python main.py
```

Then visit `http://localhost:5000`

### Running Tests

```bash
pip install -r requirements_dev.txt
pytest tests/ -v
```

---

## 🔬 MLOps Infrastructure

### Experiment Tracking (MLflow)

```bash
mlflow ui
```

Features:
- Hyperparameter tracking
- Metrics logging
- Model artifacts storage
- Experiment comparison

### Workflow Orchestration (Airflow)

```bash
airflow webserver --port 8080
airflow scheduler
```

DAGs:
- Daily model training
- Data validation
- Artifact versioning

### CI/CD Pipelines (GitHub Actions)

Automated workflows:
- ✅ Testing on Python 3.9-3.11
- ✅ Code linting (flake8)
- ✅ Docker build & push
- ✅ Daily model validation

### Monitoring & Logging

Structured logging with:
- JSON-formatted logs
- Real-time metrics collection
- Data drift detection
- Error rate tracking

---

## 🏗️ Model Architecture

**Transformer Components:**
- Multi-head self-attention (8 heads)
- Feed-forward networks
- Position-wise encodings
- Residual connections
- Layer normalization

**Hyperparameters:**
```python
d_model = 128
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 256
dropout = 0.1
seq_length = 128
```

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_model.py -v
```

---

## 🐳 Docker

```bash
# Build image
docker build -t transformer:latest .

# Run container
docker run -p 5000:5000 transformer:latest python main.py
```

---

## 📚 Documentation

- **[MLOPS_IMPLEMENTATION.md](MLOPS_IMPLEMENTATION.md)** - MLOps setup
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Architecture details
- **[QUICK_REFERENCE.sh](QUICK_REFERENCE.sh)** - Quick commands

---

## 🤝 Contributing

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and run tests
pytest tests/ -v

# Commit and push
git add -A
git commit -m "Add feature: description"
git push origin feature/my-feature
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- PyTorch team for deep learning framework
- Hugging Face for tokenizers & datasets
- Apache Airflow for orchestration
- MLflow for experiment tracking

---

**Made with ❤️ - Last Updated: January 12, 2026**
