# Generalizability VS Counterfactual Explainability


## Project Structure

```
├── main.py                 # Main entry point
├── data/
│   └── water_potability.csv    # Water quality dataset
├── src/
│   ├── dataset.py          # Data loading and preprocessing
│   ├── models.py           # Neural network architectures
│   ├── loss.py             # Loss functions
│   ├── optimizer.py        # Optimizer configurations
│   ├── estimator.py        # Monte Carlo counterfactual estimation
│   ├── geometric.py        # Hypersphere geometry utilities
│   ├── evaluation.py       # Model evaluation metrics
│   └── trainer.py          # PyTorch Lightning training module
└── log/                    # Training logs and outputs
```

## Key Components

### Models ([`src/models.py`](src/models.py))
- **Binary Logistic Regression** ([`BLogisticRegression`](src/models.py)): Simple linear classifier
- **Binary MLP** ([`BMLP`](src/models.py)): Multi-layer perceptron with configurable architecture

### Counterfactual Estimation ([`src/estimator.py`](src/estimator.py))
The [`MontecarloEstimator`](src/estimator.py) class implements:
- Random perturbation generation within hyperspheres
- Counterfactual prediction evaluation
- Robustness metrics computation

### Geometric Analysis ([`src/geometric.py`](src/geometric.py))
The [`Sphere`](src/geometric.py) class provides:
- Hypersphere volume calculations
- Uniform and normal point sampling within spheres
- Multi-dimensional geometric utilities

### Training Framework ([`src/trainer.py`](src/trainer.py))
The [`LightningClassifier`](src/trainer.py) implements:
- Integrated counterfactual estimation during training
- Margin-based generalization bounds
- Comprehensive metric logging

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd generalizability_ce_tradeoff

# Create the conda env
conda create --name ce
conda activate ce
conda install python=3.11

# Install required dependencies
pip install torch pytorch-lightning torchmetrics pandas scikit-learn scipy matplotlib numpy lightning
```

## Usage

### Command Line Arguments

The main script supports the following command-line arguments:

```bash
python main.py [OPTIONS]
```

#### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_name` | `str` | `water` | Dataset name to use for training |
| `--seed_split` | `int` | `42` | Random seed for dataset splitting |
| `--degree` | `int` | `1` | Polynomial degree for feature transformation |
| `--model_type` | `str` | `BMLP` | Model architecture (`BMLP` or `LogisticRegression`) |
| `--dropout` | `float` | `0.0` | Dropout rate for neural network layers |
| `--n_samples` | `int` | `1000` | Number of Monte Carlo samples for counterfactual estimation |
| `--radius` | `int` | `100` | Radius parameter for perturbation hypersphere |
| `--batch_size` | `int` | `128` | Batch size for training and validation |
| `--distribution` | `str` | `uniform` | Distribution type for sampling (`uniform` or `normal`) |
| `--epochs` | `int` | `500` | Maximum number of training epochs |

### Example Usage

#### Basic Training (Default Parameters)
```bash
python main.py
```

#### Logistic Regression with Custom Parameters
```bash
python main.py --model_type LogisticRegression --epochs 100 --batch_size 64
```

#### MLP with Dropout and Custom Sampling
```bash
python main.py \
    --model_type BMLP \
    --dropout 0.2 \
    --n_samples 2000 \
    --radius 50 \
    --distribution normal \
    --epochs 300
```


### Model-Specific Notes

- **BMLP Model**: When using `--model_type BMLP`, the script automatically configures a multi-layer perceptron with hidden layers `[100, 30]`
- **Logistic Regression**: When using `--model_type LogisticRegression`, dropout and hidden layer parameters are ignored
- **Margin Calculation**: Margin computation is automatically disabled for BMLP models and enabled for Logistic Regression

### Optimization Configuration

The script uses SGD optimizer with the following fixed configuration:
- **Optimizer**: SGD
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0

### Hardware Requirements

The script automatically detects and uses GPU acceleration when available:
- **Accelerator**: GPU (if available, otherwise CPU)
- **CUDA Support**: Automatic detection and device information logging

### Output and Logging

Training logs are saved to `log/{dataset_name}_{model_type}/` directory with CSV format containing:
- Training and validation metrics
- Counterfactual estimation results
- Geometric bound calculations
- Performance statistics