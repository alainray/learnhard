import comet_ml
import torch
from models import get_model
from datasets import get_dataloaders
from utils import checkpoint, get_args
from opt import get_opt, get_criterion
from train import train, test, trainBPR
from utils import setup_comet, set_random_state

# Handling parameters to experiments
args = get_args()
# Set randomness
set_random_state(args)
# Create datasets and dataloaders
train_dl, test_dls = get_dataloaders(args)
# Define model parameters
model = get_model(args)
# Define optimizer
opt = get_opt(args, model)
# Define criterion
criterion = get_criterion(args)
# Training setup
model.to(args.device)
# Comet.ml logging
exp = setup_comet(args)
exp.log_parameters({k:w for k,w in vars(args).items() if "comet" not in k})
model.comet_experiment_key = exp.get_key() # To retrieve existing experiment

# Training
for epoch in range(1, args.epochs + 1):
    if args.method == "BPR":
        model, stats = trainBPR(exp, args, model, train_dl, opt, criterion, epoch)
    else:
        model, stats = train(exp, args, model, train_dl, opt, criterion, epoch)
    checkpoint(args, model, stats, epoch, split="train")
    for i, test_dl in enumerate(test_dls):
        model, stats = test(exp, args, model, test_dl, criterion, epoch, prefix=f"test{i+1}")
        checkpoint(args, model, stats, epoch, split=f"test{i+1}")
