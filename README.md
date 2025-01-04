# VisTabNet

[![PyPI version](https://badge.fury.io/py/vistabnet.svg)](https://badge.fury.io/py/vistabnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

VisTabNet is a powerful Vision Transformer-based Tabular Data Classifier that leverages the strength of transformer architectures for tabular data classification tasks.

## Features

- Vision Transformer architecture adapted for tabular data
- Simple and intuitive API similar to scikit-learn
- GPU acceleration support
- Automatic handling of numerical features
- Built-in evaluation metrics
- Compatible with pandas DataFrames and numpy arrays

## Installation

You can install VisTabNet using pip:

```bash
pip install vistabnet
```

## Quick Start

Here's a simple example to get you started:

```python
from vistabnet import VisTabNetClassifier
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

# Prepare your data
X_train, y_train, X_test, y_test = ... # Load your data here
# Note: y should be label encoded, not one-hot encoded

# Initialize the model
model = VisTabNetClassifier(
    input_features=X_train.shape[1],
    classes=len(np.unique(y_train)),
    device="cuda"  # Use "cpu" if no GPU is available
)

# Train the model
model.fit(
    X_train,
    y_train,
    eval_X=X_test,
    eval_y=y_test
)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced accuracy: {accuracy}")
```

## Advanced Usage

### Model Configuration

You can customize the VisTabNet model by adjusting various parameters:

```python
model = VisTabNetClassifier(
    input_features=X_train.shape[1],
    classes=len(np.unique(y_train)),
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    device="cuda"
)
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision ≥ 0.15.0
- tqdm ≥ 4.65.0
- focal-loss-torch ≥ 0.1.2

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use VisTabNet in your research, please cite:

```bibtex
@misc{wydmański2024vistabnetadaptingvisiontransformers,
      title={VisTabNet: Adapting Vision Transformers for Tabular Data}, 
      author={Witold Wydmański and Ulvi Movsum-zada and Jacek Tabor and Marek Śmieja},
      year={2024},
      eprint={2501.00057},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.00057}, 
}
```

## Support

For questions and support, please open an issue in the GitHub repository.