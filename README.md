# Oil Well Production Forecasting Model

## Overview
This project implements a deep learning model for forecasting oil well production rates using a combination of temporal and spatial features. The model utilizes both transformer architecture and graph neural networks to capture complex relationships between wells and their production patterns.

## Features
- Multi-well time series forecasting
- Spatial relationship modeling between wells
- Transformer-based temporal processing
- Graph neural network for spatial dependencies
- Support for pressure and rate predictions
- Data preprocessing and scaling capabilities

## Requirements
Main dependencies:
- torch>=2.5.1
- torch-geometric>=2.6.1
- pytorch-lightning>=2.4.0
- pandas>=2.2.3
- numpy>=2.1.3
- matplotlib>=3.9.2
- tsl>=2.0

For complete dependencies, see `requirements.txt`.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
- `train.py` - Main training script and model definitions
- `evaluate.py` - Evaluation and visualization utilities
- `mod.py` - Data modification and preprocessing utilities

## Usage

1. Data Preparation:
   ```bash
   python mod.py
   ```

2. Training:
   ```bash
   python train.py
   ```

3. Evaluation:
   ```bash
   python evaluate.py
   ```

## Model Architecture
The model combines two main components:
1. Transformer-based temporal processing
2. Graph Neural Network for spatial relationships

Key parameters:
- Input window: 3 time steps
- Prediction horizon: 12 time steps
- Hidden size: 32
- Number of transformer layers: 2

## Data Format
Input data should be in CSV format with the following columns:
- Well Name
- Date
- Production rates
- Pressure measurements
- Well coordinates (X, Y)

## License
MIT

## Authors
[Your Name]

## Acknowledgments
- TSL (Torch Spatiotemporal) library
- PyTorch Geometric
