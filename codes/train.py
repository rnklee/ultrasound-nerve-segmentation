import sys
import torch
import numpy as np
import pandas as pd
import cv2
from loss import *
from utils import *
from functools import partial
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class NerveTrainset(Dataset):
    def __init__(self, mskfiles, path, preprocess=None, augment=None, train=True):
        super().__init__()

        self.mskfiles = [path+o for o in mskfiles]
        self.imgfiles = [o.replace('_mask', '') for o in self.mskfiles]

        self.preprocess = preprocess
        self.augment = augment


    def __len__(self):
        return len(self.mskfiles)


    def __getitem__(self, idx):
        img = cv2.imread(self.imgfiles[idx], cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(self.mskfiles[idx], cv2.IMREAD_GRAYSCALE)

        if self.augment is not None:
            sample = self.augment(image=img, mask=msk)
            img, msk = sample['image'], sample['mask']

        if self.preprocess is not None:
            sample = self.preprocess(image=img, mask=msk)
            img, msk = sample['image'], sample['mask']

        # img = np.expand_dims(img, axis=0)
        msk = np.expand_dims(msk, axis=0)

        img = img.astype(np.float32)
        msk = (msk/255).astype(np.float32)

        return img, msk



### Adopted from smp
### https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/utils
class Meter(object):
    """
    Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


### Adopted from smp
### https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/utils
class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            #self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n) this doesn't seem right...
            self.mean = float(self.sum)/float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


### Adopted from smp
### https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/utils
class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        batch_count = 0

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                #batch_size = x.shape[0]
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


### Adopted from smp
### https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/utils
class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        #prediction = self.model.forward(x)
        prediction = self.model(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


### Adopted from smp
### https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/utils
class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            #prediction = self.model.forward(x)
            prediction = self.model(x)
            loss = self.loss(prediction, y)
        return loss, prediction


def train_validate(train_msks, valid_msks, path, preprocess, augment, model, optim,
                   loss=DiceLoss(soft=True), metrics=[DiceLoss(soft=True), DiceLoss(soft=False)],
                   device='cuda', batch_size=16, epochs=5,
                   save_thres=1e6, save_metric='hard_dice', save_path=None, stop_early=False,
                   verbose=True):

    train_nerves = np.array([has_nerve(o, path) for o in train_msks])
    train_no_nerves, train_nerves = np.where(train_nerves==0), np.where(train_nerves>0)
    train_nerve_ratio = len(train_nerves[0])/len(train_msks)
    print(f'{len(train_msks)} samples in train set: {len(train_nerves[0])} with nerves {len(train_no_nerves[0])} with no nerve.')
    print(f'Ratio of samples with nerves in train set: {train_nerve_ratio:.4}')

    valid_nerves = np.array([has_nerve(o, path) for o in valid_msks])
    valid_no_nerves, valid_nerves = np.where(valid_nerves==0), np.where(valid_nerves>0)
    nerve_ratio = len(valid_nerves[0])/len(valid_msks)
    print(f'{len(valid_msks)} samples in validation set: {len(valid_nerves[0])} with nerves {len(valid_no_nerves[0])} with no nerve.')
    print(f'Ratio of samples with nerves in validation set: {nerve_ratio:.4}')
    valid_msks0, valid_msks1 = np.array(valid_msks)[valid_no_nerves], np.array(valid_msks)[valid_nerves]

    traindata = NerveTrainset(mskfiles=train_msks, path=path, preprocess=preprocess, augment=augment)
    validdata0 = NerveTrainset(mskfiles=valid_msks0, path=path, preprocess=preprocess)
    validdata1 = NerveTrainset(mskfiles=valid_msks1, path=path, preprocess=preprocess)
    validdata = NerveTrainset(mskfiles=valid_msks, path=path, preprocess=preprocess)

    trainloader = DataLoader(traindata, batch_size = batch_size, shuffle = True)
    validloader0 = DataLoader(validdata0, batch_size = 1)
    validloader1 = DataLoader(validdata1, batch_size = 1)
    validloader = DataLoader(validdata, batch_size = 1)

    trainepoch = TrainEpoch(model, optimizer=optim, device=device, loss=loss, metrics=metrics, verbose=verbose)
    validepoch = ValidEpoch(model, device=device, loss=loss, metrics=metrics, verbose=verbose)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)

    if stop_early:
        stopper = EarlyStopper()

    columns = ['epoch', 'type'] + [loss.__name__] +  [metric.__name__ for metric in metrics]
    loss_log = pd.DataFrame(columns=columns)
    for i in tqdm(range(epochs), disable=verbose):
        if verbose:
            print('\nEpoch: {}'.format(i))

        trainlogs = trainepoch.run(trainloader)
        validlogs0 = validepoch.run(validloader0)
        validlogs1 = validepoch.run(validloader1)
        # validlogs = validepoch.run(validloader)

        train_loss = {'epoch': i, 'type': 'train'}
        valid_loss0 = {'epoch': i, 'type': 'valid_no_nerve'}
        valid_loss1 = {'epoch': i, 'type': 'valid_nerve'}
        valid_loss = {'epoch': i, 'type': 'valid'}

        train_loss.update(dict([(loss.__name__, trainlogs[loss.__name__])]))
        valid_loss0.update(dict([(loss.__name__, validlogs0[loss.__name__])]))
        valid_loss1.update(dict([(loss.__name__, validlogs1[loss.__name__])]))
        valid_loss.update(dict([(loss.__name__, validlogs0[loss.__name__]*(1-nerve_ratio) + validlogs1[loss.__name__]*(nerve_ratio))]))

        train_loss.update(dict([(metric.__name__, trainlogs[metric.__name__]) for metric in metrics]))
        valid_loss0.update(dict([(metric.__name__, validlogs0[metric.__name__]) for metric in metrics]))
        valid_loss1.update(dict([(metric.__name__, validlogs1[metric.__name__]) for metric in metrics]))
        valid_loss.update(dict([(metric.__name__, validlogs0[metric.__name__]*(1-nerve_ratio) + validlogs1[metric.__name__]*(nerve_ratio)) for metric in metrics]))
        
        valid_loss_msg = ['{} - {:.4}'.format(metric.__name__, valid_loss[metric.__name__]) for metric in [loss] + metrics]
        valid_loss_msg = ', '.join(valid_loss_msg)
        if verbose:
            print('validation: ' + valid_loss_msg)

        loss_log.loc[len(loss_log)] = train_loss
        loss_log.loc[len(loss_log)] = valid_loss0
        loss_log.loc[len(loss_log)] = valid_loss1
        loss_log.loc[len(loss_log)] = valid_loss

        save_loss = valid_loss[save_metric]
        if save_path is not None and save_loss < save_thres:
            save_thres = save_loss
            torch.save(model, save_path)

        if stop_early:
            if stopper.early_stop(valid_loss[save_metric]):
                print('Early stop triggered.')
                break
            print(f'Valid Loss: {valid_loss[save_metric]:.4}, Stopper: min_validation_loss {stopper.min_validation_loss:.4}, counter {stopper.counter}.')

        lr_scheduler.step()

    return loss_log


def make_single_pred(model, img, preprocess, path=None, from_file=False, device='cuda', eval=True):
    if eval:
        model.eval()
    else:
        model.train()

    if from_file:
        img = cv2.imread(path+img, cv2.IMREAD_GRAYSCALE)
    if preprocess is not None:
        img = preprocess(image=img)['image']
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    pred = model(img)
    return pred.cpu().detach().numpy().squeeze()



@torch.no_grad()
def predict_compare(mskfiles, path, model, preprocess=None, postprocess=None,
                    metrics=[DiceLoss(soft=False), IoULoss(soft=False), TargetSize(), OutputSize()], device='cpu'):
    if isinstance(model, str):
        test_model = torch.load(model)
    else: test_model = model
    model.eval()

    test_data = NerveTrainset(mskfiles=mskfiles, path=path, preprocess=preprocess)
    test_loader = DataLoader(test_data, batch_size=1) # DataLoader probably unnecessary here. Still a good practice?

    # columns = [metric.__name__ for metric in metrics] + ['gt_mask_size', 'pred_mask_size']
    columns = [metric.__name__ for metric in metrics]
    test_log = pd.DataFrame(columns=columns)
    with tqdm(
        test_loader,
        disable=False
    ) as iterator:
        for x, y in iterator:
            x, y = x.to(device), y.to(device)

            # with torch.no_grad():
            #     pred = test_model(x)
            pred = test_model(x)
            if postprocess is not None:
                pred = postprocess(pred)
            else:
                pred = pred.round()

            # update the log
            log_update = {metric.__name__: metric(pred,y).item() for metric in metrics}
            # log_update.update({'gt_mask_size': y.sum().item(), 'pred_mask_size': pred.sum().item()})
            test_log.loc[len(test_log)] = log_update

    return test_log