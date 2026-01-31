
# ddms: Devesh's Deep Model Suite

**ddms** is a modular deep learning training system built for flexibility, reproducibility, and rapid experimentation.

## Key Features

- 🔌 Plug-and-play models: Custom, cloned, or Hugging Face
- 🧮 Custom loss and optimizer support
- 📝 YAML config + CLI override (CLI always wins)
- 📊 Weights & Biases (wandb) logging with custom run names
- 🛑 Always prints config and asks for confirmation before running

---

## How to Use ddms

### 1. Add Your Model
- Put your model in `dl/models/custom/YourModel.py` (or `cloned/` for external code)
- Inherit from `torch.nn.Module`
- See `dl/models/template_model.py` for a starting point

### 2. Configure Your Experiment
- Edit `dl/train/config.yaml`:
	- Set model source/name and parameters
	- Choose loss function and its params
	- Set optimizer type/params
	- Set dataset and run settings
	- Set wandb project and plot name

### 3. Run Training
- Override any config option from the CLI:
	```bash
	python dl/train/train.py --model__source custom --model__name YourModel --loss__type focal --wandb__plot_name my_exp --lr 0.001
	```
- The script prints the config and asks for y/n before starting

### 4. Track Experiments with wandb
- Run `wandb login` once or set your API key
- All runs are logged to wandb with your chosen plot name

---


## Example ddms Config (`dl/train/config.yaml`)

```yaml
model:
	source: custom
	name: vit
	depth: 10
	in_dim: 192
	inner_dim: 128
	num_classes: 10
	img_size: 32
	patch_size: 8
	in_channels: 3

loss:
	type: devesh_loss   # or cross_entropy, focal, etc.
	params:
		alpha: 0.5

optimizer:
	type: adam
	params: {"weight_decay": 0.01}

wandb:
	project: ddms-training
	plot_name: devesh_exp_1

dataset:
	name: cifar10
	path: ../data/cifar-10-batches-py

run:
	batch_size: 32
	lr: 0.0001
	epochs: 10
	save_path: "./checkpoint/cifar_transformer.pth"
	device: "cpu"
	quantize: false
	quantized_save_path: "./checkpoint/model_quantized.pth"
```

---


## Custom Loss & Optimizer in ddms

- **Custom loss functions** must subclass `torch.nn.Module` and implement `forward`.
- Add your loss to `dl/utils/loss.py` and register it in `get_loss_fn` in the trainer.
- Example:
	```python
	from torch import nn
	import torch
	class DeveshLoss(nn.Module):
			def __init__(self, alpha=1.0):
					super().__init__()
					self.alpha = alpha
			def forward(self, pred, target):
					return torch.mean((pred - target) ** 2) * self.alpha
	```
- In config:
	```yaml
	loss:
		type: devesh_loss
		params:
			alpha: 0.5
	```
- Register in trainer:
	```python
	def get_loss_fn(loss_cfg):
			if loss_cfg['type'] == 'devesh_loss':
					from utils.loss import DeveshLoss
					return DeveshLoss(**loss_cfg.get('params', {}))
	```
- **Custom optimizer:**
	- In config:
		```yaml
		optimizer:
			type: adam
			params: {"weight_decay": 0.01}
		```
	- Or via CLI:
		```bash
		python dl/train/train.py --optimizer__type sgd --optimizer__params '{"momentum": 0.9}'
		```


## Directory Structure

- `dl/models/custom/` — Your custom models
- `dl/models/cloned/` — Models cloned from external sources
- `dl/utils/` — Trainer, loss functions, dataloaders
- `dl/train/` — Training scripts and configs


## Example: Add a Custom Model

1. Create `dl/models/custom/DeveshNet.py`:
   ```python
   import torch.nn as nn
   class DeveshNet(nn.Module):
	   def __init__(self, ...):
		   super().__init__()
		   # ...
	   def forward(self, x):
		   # ...
   ```
2. Set `model.source: custom` and `model.name: DeveshNet` in config or CLI.


## Example: Use a Hugging Face Model

1. Install transformers:
	```bash
	pip install transformers
	```
2. Set `model.source: huggingface` and `model.name: <hf-model-id>` in config or CLI.


## Notes
- All config options can be set in YAML or overridden via CLI (e.g., `--loss__type devesh_loss`).
- The trainer is fully modular and will prompt for confirmation before running.
- For wandb, set your API key once with `wandb login` or via `WANDB_API_KEY` env variable.
