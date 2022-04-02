from comet_ml import Experiment, ExistingExperiment
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import os
from os import mkdir
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
from functools import wraps
from time import time


# Comet Experiments
def setup_comet(args, resume_experiment_key=''):
    api_key = args.cometKey # w4JbvdIWlas52xdwict9MwmyH
    workspace = args.cometWs # alainray
    project_name = args.cometName # learnhard
    enabled = bool(api_key) and bool(workspace)
    disabled = not enabled

    print(f"Setting up comet logging using: {{api_key={api_key}, workspace={workspace}, enabled={enabled}}}")

    if resume_experiment_key:
        experiment = ExistingExperiment(api_key=api_key, previous_experiment=resume_experiment_key)
        return experiment

    experiment = Experiment(api_key=api_key, parse_args=False, project_name=project_name,
                            workspace=workspace, disabled=disabled)
    # TEST
    experiment_name = get_prefix(args)
    if experiment_name:
        experiment.set_name(experiment_name)

    train_data_type = os.environ.get('TRAIN_DATA_TYPE')
    if train_data_type:
        experiment.add_tag(train_data_type)

    tags = os.environ.get('TAGS')
    if tags:
        experiment.add_tags(tags.split(','))

    return experiment



def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"Func: {f.__name__} took: {te-ts:0.0f} sec")
        return result
    return wrap

@timing
def train(experiment, args, model, loader, opt, device, criterion, epoch):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.train()
    acc_data = ""
    total_batches = len(loader)
    metrics = {'loss': None, 'acc': None}
    # confusion matrix
    y_pred = []
    y_true = []

    for n_batch, (index, x, label) in enumerate(loader):
        opt.zero_grad()
        x = x.to(device)
        bs = x.shape[0]
        label = label.to(device)
        logits = model(x)

        if args.label_type == "score":
            label = label.float().unsqueeze(1)
        else:
            preds = logits.argmax(dim=1)
            y_pred.extend(preds.detach().cpu().numpy())
            y_true.extend(label.detach().cpu().numpy())
            correct = (preds == label).cpu().sum()
            acc_meter.update(correct.cpu() / float(bs), bs)  
            acc_data = f"Acc: {100 * correct.float() / bs:.1f}% Cum. Acc: {100 * acc_meter.avg:.1f}%"
            metrics['acc'] = float(100*acc_meter.avg)
        loss = criterion(logits, label)
        loss.backward()
        # Update stats
        opt.step()
        cur_loss = loss.detach().cpu()
        metrics['loss'] = float(cur_loss)
        loss_meter.update(cur_loss, bs)
        loss_data = f" Loss (Current): {cur_loss:.3f} Cum. Loss: {loss_meter.avg:.3f}"
        training_iteration = total_batches*(epoch-1) + n_batch + 1
        experiment.log_metrics(metrics, prefix='train', step=training_iteration, epoch=epoch)

        print(f"\r[TRAIN] Epoch {epoch}: {n_batch + 1}/{total_batches}: {loss_data} {acc_data}", end="", flush=True)
    experiment.log_confusion_matrix(y_true,
                                    y_pred,
                                    step=epoch, 
                                    title=f"Confusion Matrix TRAIN, Epoch {epoch}",
                                    file_name=f"cf_{get_prefix(args)}_train_{epoch}.json")
    
    return model, [loss_meter, acc_meter]

@timing
def test(experiment, args, model, loader, device, criterion, epoch, prefix="test"):

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.eval()
    acc_data = ""
    total_batches = len(loader)
    metrics = {'loss': None, 'acc': None}
    # confusion matrix
    y_pred = []
    y_true = []
    with torch.no_grad():
        for n_batch, (index, x, label) in enumerate(loader):
            x = x.to(device)
            bs = x.shape[0]
            label = label.to(device)
            logits = model(x)

            if args.label_type == "score":
                label = label.float().unsqueeze(1)
            else:
                preds = logits.argmax(dim=1)
                y_pred.extend(preds.detach().cpu().numpy())
                y_true.extend(label.detach().cpu().numpy())
                correct = (preds == label).cpu().sum()
                acc_meter.update(correct.cpu() / float(bs), bs)  
                acc_data = f"Acc: {100 * correct.float() / bs:.1f}% Cum. Acc: {100 * acc_meter.avg:.1f}%"
                metrics['acc'] = float(100*acc_meter.avg)
            loss = criterion(logits, label)
            # Update stats
            cur_loss = loss.detach().cpu()
            metrics['loss'] = float(cur_loss)
            loss_meter.update(cur_loss, bs)
            loss_data = f" Loss (Current): {cur_loss:.3f} Cum. Loss: {loss_meter.avg:.3f}"
            training_iteration = total_batches*(epoch-1) + n_batch + 1
            experiment.log_metrics(metrics, prefix=prefix, step=training_iteration, epoch=epoch)

            print(f"\r[{prefix.upper()}] Epoch {epoch}: {n_batch + 1}/{total_batches}: {loss_data} {acc_data}", end="", flush=True)
    experiment.log_confusion_matrix(y_true,
                                    y_pred,
                                    step=epoch, 
                                    title=f"Confusion Matrix {prefix.upper()}, Epoch {epoch}",
                                    file_name=f"cf_{get_prefix(args)}_{prefix}_{epoch}_.json")
    return model, [loss_meter, acc_meter]


def create_splits(f, s, l, root="c_score/imagenet"):
    d= {'train': {'filenames': None, 'scores': None, 'labels': None},
            'test': {'filenames': None, 'scores': None, 'labels': None}}
    
    f_tr, f_ts, s_tr, s_ts, l_tr, l_ts = train_test_split(f,s,l, test_size=0.2, random_state=123)
    
    d['train']['filenames'] = f_tr
    d['train']['scores'] = s_tr
    d['train']['labels'] = l_tr
    d['test']['filenames'] = f_ts
    d['test']['scores'] = s_ts
    d['test']['labels'] = l_ts

    print("Armando archivos...")
    for split, v in d.items():
        for name in v.keys():
            np.save(join(root,f"{name}_{split}.npy"), d[split][name])

def get_prefix(args):
    return "_".join([str(w) for k,w in vars(args).items() if "comet" not in k])
def checkpoint(args, model, stats, epoch, root="ckpts", res_root="stats", split="train"):
    
    prefix = get_prefix(args)
    
    if split == "train":
        if not os.path.isdir(root):
            mkdir(root)

        torch.save(model.state_dict(), f"{root}/{prefix}_{split}_{epoch}.pth")
        
    if not os.path.isdir(res_root):
        mkdir(res_root)
    torch.save(stats, f"{res_root}/{prefix}_{split}_{epoch}.pth")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    root = "c_score/cifar10"
    #filenames = np.load(join(root,"scores.npy"), allow_pickle=True)
    filenames = np.array(list(range(50000)))
    scores = np.load(join(root,"scores.npy"), allow_pickle=True)
    labels = np.load(join(root,"labels.npy"), allow_pickle=True)
    create_splits(filenames, scores, labels, root=root)