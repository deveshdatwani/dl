How to use ddms [Devesh's Deep Model Suite]

How to use ddms

1 put your model in dl/models/custom/YourModel.py or in cloned for external code your model must inherit from torch.nn.Module see dl/models/template_model.py for a starting point

2 edit dl/train/config.yaml set model source and name and parameters set loss type and params set optimizer type and params set dataset and run settings set wandb project and plot name

3 run training you can override any config option from the command line for example python dl/train/train.py --model__source custom --model__name YourModel --loss__type focal --wandb__plot_name my_exp --lr 0.001 the script will print the config and ask for y or n before starting

4 to use wandb run wandb login once or set your api key all runs are logged to wandb with your chosen plot name

example config for dl/train/config.yaml

model source custom name vit depth 10 in_dim 192 inner_dim 128 num_classes 10 img_size 32 patch_size 8 in_channels 3
loss type devesh_loss params alpha 0.5
optimizer type adam params weight_decay 0.01
wandb project ddms-training plot_name devesh_exp_1
dataset name cifar10 path ../data/cifar-10-batches-py
run batch_size 32 lr 0.0001 epochs 10 save_path ./checkpoint/cifar_transformer.pth device cpu quantize false quantized_save_path ./checkpoint/model_quantized.pth

to add a custom loss function make a class in dl/utils/loss.py that inherits from torch.nn.Module and implements forward for example class DeveshLoss(nn.Module): def __init__(self, alpha=1.0): super().__init__() self.alpha = alpha def forward(self, pred, target): return torch.mean((pred - target) ** 2) * self.alpha then in config set loss type devesh_loss and params alpha 0.5 register your loss in get_loss_fn in the trainer

to use a custom optimizer set optimizer type and params in config or use the command line for example python dl/train/train.py --optimizer__type sgd --optimizer__params '{"momentum": 0.9}'

directory structure
dl/models/custom your custom models dl/models/cloned models cloned from external sources dl/utils trainer loss functions dataloaders dl/train training scripts and configs

to add a custom model make a file in dl/models/custom for example DeveshNet.py with a class DeveshNet(nn.Module): ... set model.source to custom and model.name to DeveshNet in config or cli

to use a huggingface model install transformers with pip install transformers set model.source to huggingface and model.name to the huggingface model id in config or cli

all config options can be set in yaml or overridden from the command line the trainer will always print the config and ask for confirmation before running for wandb set your api key once with wandb login or with the WANDB_API_KEY environment variable
	depth: 10
