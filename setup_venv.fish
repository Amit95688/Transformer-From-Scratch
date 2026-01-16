#!/usr/bin/env fish

echo "Setting up virtual environment for Transformer project..."

# Create virtual environment if it doesn't exist
if not test -d .venv
    echo "Creating virtual environment..."
    python3 -m venv .venv
end

# Activate using fish-compatible method
echo "Installing packages..."
.venv/bin/pip install --upgrade pip

echo "Installing project dependencies (minimal)..."
.venv/bin/pip install -r requirements_minimal.txt

echo "Installing project as editable package..."
.venv/bin/pip install -e .

echo ""
echo "✓ Setup complete!"
echo ""
echo "To run the training pipeline:"
echo "  .venv/bin/python src/TransformerModel/pipelines/training_pipeline.py"
echo ""
echo "Or activate the environment (in fish shell):"
echo "  source .venv/bin/activate.fish"
echo "  python src/TransformerModel/pipelines/training_pipeline.py"
