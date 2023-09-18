import sys
import torch
import cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm
from google.colab import files
from torch.utils.data import Dataset, DataLoader


class NerveTestset(Dataset):
    def __init__(self, imgfiles, path, preprocess=None):
        super().__init__()

        self.imgfiles = [path+o for o in imgfiles]
        self.path = path
        self.preprocess = preprocess


    def __len__(self):
        return len(self.imgfiles)


    def __getitem__(self, idx):
        img = cv2.imread(self.imgfiles[idx], cv2.IMREAD_GRAYSCALE)

        if self.preprocess is not None:
            sample = self.preprocess(image = img)
            img = sample['image']

        img = img.astype(np.float32)
        return img


    def get_file(self, idx):
        return self.imgfiles[idx]


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()>0)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b

    encoding = ' '.join(map(str, run_lengths))

    return encoding


def default_postprocess(pred, dsize=(580,420)):
    pred = cv2.resize(pred, dsize, cv2.INTER_NEAREST)
    pred = pred.round()
    return pred


@torch.no_grad()
def predict_test(test_imgs, path, model, preprocess=None, postprocess=None, device='cuda'):
    if isinstance(model, str):
        test_model = torch.load(model)
    else:
        test_model = model

    model.eval()

    test_data = NerveTestset(imgfiles=test_imgs, path=path, preprocess=preprocess)
    test_loader = DataLoader(test_data, batch_size=1) # DataLoader probably unnecessary here. Still a good practice?

    test_preds = []
    with tqdm(
        test_loader,
        disable=False
    ) as iterator:
        for x in iterator:
            x = x.to(device)

            # with torch.no_grad():
            #     pred = test_model(x)
            pred = test_model(x).cpu().detach().numpy()
            pred = pred.squeeze()

            if postprocess is None:
                pred = default_postprocess(pred)
            else:
                for fn in postprocess:
                    pred = fn(pred)

            test_preds.append(pred)

    return test_preds



def create_submission(test_imgs, path, model, preprocess, postprocess=None, filename=None, device='cuda'):
    test_preds = predict_test(test_imgs, path, model, preprocess, postprocess, device)

    create_csv(test_preds, filename)

    return test_preds



def create_csv(preds, filename=None):
    if filename is None:
        filename = 'submission' + datetime.now().strftime('_%m%d%H%M') +'.csv'

    first_row = 'img,pixels'
    with open(filename, 'w+') as f:
        f.write(first_row + '\n')
        for i, pred in enumerate(preds):
            # s = str(test_imgs[i].replace('.tif', '')) + ',' + rle_encoding(pred)
            s = str(int(i+1)) + ',' + rle_encoding(pred)
            f.write(s + '\n')

    files.download(filename)