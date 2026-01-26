# DL

Modular Deep Learning Framework
==============================

## Structure

- `models/` — All model architectures (ViT, Edter, etc.)
- `train/` — Training scripts and configs
- `utils/` — Data loaders, loss functions, trainer utilities
- `tests/` — Unit tests for models and training

## Getting Started

1. Install requirements:
	```bash
	pip install -r requirements.txt
	```
2. Train a model:
	```bash
	python train/train.py --model vit --dataset cifar10 --config train/config.yaml
	```

## Adding a New Model

1. Create a new file in `models/` (see `template_model.py` for example).
2. Add your model to `models/__init__.py` and `MODEL_REGISTRY` in `train/train.py`.
3. Add a test in `tests/`.

## Configuration

Edit `train/config.yaml` to change model parameters, training settings, etc.

## Testing

Run all tests:
```bash
pytest tests/
```

## License

MIT License
