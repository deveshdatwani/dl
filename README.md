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
3. (Optional) Enable post-training quantization:
	- Add `quantize: true` to your `train/config.yaml`.
	- Optionally set `quantized_save_path: ./checkpoint/model_quantized.pth`.

## Post-Training Optimization

You can apply post-training quantization to reduce model size and improve inference speed:

- Dynamic quantization is supported out of the box. After training, if `quantize: true` is set in your config, a quantized model will be saved.
- See `optim/post_training.py` for more utilities (static quantization, etc).

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
