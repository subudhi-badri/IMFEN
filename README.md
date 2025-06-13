# Multi-Feature Enhancement Network (MFEN)

This repository contains the implementation of a Multi-Feature Enhancement Network for underwater image enhancement. The model uses a combination of dense blocks, attention mechanisms, and multi-scale feature processing to enhance underwater images.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mfen.git
cd mfen
```
Use python 3.11


2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
mfen/
├── mfen/              # Main package directory
│   ├── models/        # Model architectures
│   ├── datasets/      # Dataset implementations
│   ├── losses/        # Loss functions
│   └── utils/         # Utility functions
├── configs/           # Configuration files
└── requirements.txt   # Project dependencies
```

## Usage

### Training

To train the model:

```bash
python -m mfen.train --config configs/default_config.py
```

### Evaluation

To evaluate the model:

```bash
python -m mfen.train --config configs/default_config.py --mode eval
```

## Model Architecture

The MFEN model consists of:
- Encoder with RRDB (Residual in Residual Dense Block) and SAM (Scale Attention Module)
- Decoder with multi-scale feature fusion
- Attention mechanisms for feature refinement

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mfen2024,
  title={Multi-Feature Enhancement Network for Underwater Image Enhancement},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
``` 