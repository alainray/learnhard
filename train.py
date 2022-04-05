import torch
from utils import timing, AverageMeter, get_prefix
from torch.nn import LogSigmoid, Sigmoid

@timing
def train(experiment, args, model, loader, opt, criterion, epoch):
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
        x = x.to(args.device)
        bs = x.shape[0]
        label = label.to(args.device)
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
    
    if args.label_type != "score":
        experiment.log_confusion_matrix(y_true,
                                    y_pred,
                                    step=epoch, 
                                    title=f"Confusion Matrix TRAIN, Epoch {epoch}",
                                    file_name=f"cf_{get_prefix(args)}_train_{epoch}.json")
    
    return model, [loss_meter, acc_meter]

@timing
def test(experiment, args, model, loader, criterion, epoch, prefix="test"):

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
            x = x.to(args.device)
            bs = x.shape[0]
            label = label.to(args.device)
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

@timing
def trainBPR(experiment, args, model, loader, opt, criterion, epoch):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.train()
    acc_data = ""
    total_batches = len(loader)
    metrics = {'loss': None, 'acc': None}

    for n_batch, (index, x, label) in enumerate(loader):
        opt.zero_grad()
        x = x.to(args.device)
        bs = x.shape[0]
        label = label.to(args.device)
        logits = model(x)
        losses, accs = criterion(logits, label)
        acc_meter.update(accs['batch'] / float(accs['nbatch']), accs['nbatch'])  
        acc_data = f"Acc: {100 * float(accs['batch'])/accs['nbatch']:.1f}% Cum. Acc: {100 * acc_meter.avg:.1f}%"
        acc_data += f"Correct: {accs['batch']} Total: {accs['nbatch']} Selected: {accs['nselected']}"
        metrics['acc'] = float(100*acc_meter.avg)

        loss = losses['batch']
        loss.backward()
        # Update stats
        opt.step()
        cur_loss = loss.detach().cpu()
        metrics['loss'] = float(cur_loss)
        loss_meter.update(cur_loss, accs['nbatch'])
        loss_data = f" Loss (Current): {cur_loss:.3f} Cum. Loss: {loss_meter.avg:.3f}"
        training_iteration = total_batches*(epoch-1) + n_batch + 1
        experiment.log_metrics(metrics, prefix='train', step=training_iteration, epoch=epoch)

        print(f"\r[TRAIN] Epoch {epoch}: {n_batch + 1}/{total_batches}: {loss_data} {acc_data}", end="", flush=True)
     
    return model, [loss_meter, acc_meter]

def BPRLoss(logits, labels, n = None):
    ls = LogSigmoid()
    s = Sigmoid()
    losses = {'batch': None, 'selected': None}
    acc = {'batch': None, 'selected': None}
    logits = logits.squeeze()
    labels = labels.squeeze()
    # Create all pairs between logits, then between all labels
    comb_logits = torch.combinations(s(logits),2)
    comb_labels = torch.combinations(labels,2)
    
    x = comb_logits[:,0] - comb_logits[:,1]
    if n is None:
        n = int(comb_logits.shape[0])

    sign_fix = torch.sign(comb_labels[:,0] - comb_labels[:,1])
    loss = -x*sign_fix
    #print(loss)
    #print(ls(loss))
    loss, selected = loss.sort(descending=True)
    correct = (torch.sign(x) == sign_fix).detach().cpu()

    #print(loss)
    
    acc['batch'] = int(correct.sum())
    acc['nbatch'] = int(loss.numel())
    #correct = correct[loss>0]
    #loss = loss[loss>0]
    #print(loss)
    losses['batch'] = loss.mean()
    acc['nselected'] = int(loss.numel())
    return losses, acc

if __name__ == "__main__":
    torch.manual_seed(123)
    bs = 10
    scale = 1
    a = scale*torch.randn((bs,1)).float()
    b = scale*torch.randn((bs,1)).float()

    print(BPRLoss(a, b, n = 10))

