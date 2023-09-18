import re
import random
import cv2
import numpy as np


### For data retrieval and checking
def get_subject(filename):
    reg = re.compile('[0-9]+')
    return int(reg.search(filename).group())


def get_id(filename):
    reg = re.compile('_[0-9]+')
    return int(reg.search(filename).group().replace('_', ''))


def get_mskfile(subject:int, id:int, path:str):
    return path + str(subject) + '_' + str(id) + '_mask.tif'


def get_imgfile(subject:int, id:int, path:str):
    return path + str(subject) + '_' + str(id) + '.tif'


def get_msk(subject:int, id:int, path:str):
    mskfile = get_mskfile(subject, id, path)
    return cv2.imread(mskfile, cv2.IMREAD_GRAYSCALE)


def get_img(subject:int, id:int, path:str, color=False):
    imgfile = get_imgfile(subject, id, path)
    if color: return cv2.imread(imgfile)
    return cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)


def has_nerve(file, path):
    if '_mask' in file:
        o = path+file
    else:
        o = path+file.replace('.tif', '_mask.tif')
    return 1 if cv2.imread(o, cv2.IMREAD_GRAYSCALE).any() else 0



### For preprocessing pipelines
def add_channels(img, n_channels=3, **kwargs):
    ''' Take a SINGLE channel image and increase n_channels.'''
    return np.stack([img]*n_channels, axis=2)


def apply_clahe(img, **kwargs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img.astype('uint8'))


def to_tensor(x, **kwargs):
    return x.transpose(2,0,1).astype('float32')



### For train/validation splits or cross validations
def split_subjectwise(num_dict, num=[5,7], thres=[0.4,0.5], size=[0.1,0.15], pick_from=list(range(1,48))):
    def condition_met(subjects):
        total, mask_total = 0, 0
        for sub in subjects:
            n,m = num_dict[sub]
            total += n
            mask_total += m
        ratio = mask_total/total
        return thres[0] <= ratio and ratio <= thres[1]

    train_subjects = []
    valid_subjects = []

    cond = False
    attempts = 100000
    while not cond:
        attempts -= 1
        if attempts < 0:
            return False

        valid_num = random.randint(num[0],num[1])
        valid_subjects = random.sample(pick_from, valid_num)
        train_subjects = [sub for sub in list(range(1,48)) if sub not in valid_subjects]
        #pdb.set_trace()
        size_cond = size[0] < len(valid_subjects)/len(train_subjects) and len(valid_subjects)/len(train_subjects) < size[1]
        mask_cond = condition_met(valid_subjects) and condition_met(train_subjects)
        cond = mask_cond and size_cond

    return train_subjects, valid_subjects



def fold_subjectwise(num_dict, n_fold):
    fold_attempts = 100000
    found = False
    while not found:
        if fold_attempts < 0:
            print('Failed to find a fold')
            return False

        used = []
        folds = []
        n = 0
        while n < n_fold:
            split_attempts = 100000
            pick_from = [sub for sub in list(range(1,48)) if sub not in used]
            picked = split_subjectwise(num_dict, pick_from)
            if picked:
                used += picked[1]
                folds.append(picked)
                if n == n_fold-1:
                    found = True
            else:
                split_attempts -= 1
                break
    return folds


### Postprocessing and ensembling
def restore_size(pred):
    dsize = (580,420)
    pred = cv2.resize(pred, dsize, cv2.INTER_NEAREST)
    pred = pred.round()
    return pred


def threshold(output, thres):
    output[output>=thres] = 1.
    output[output<thres] = 0.
    return output


def quantile_threshold(output, q=0.98, min_thres=0.1, max_thres=0.35):
    thres = torch.quantile(output, q)
    thres = max(thres, min_thres)
    thres = min(thres, max_thres)
    output = threshold(output, thres)
    return output


def filter_small(output, thres):
    if output.sum() < thres:
        return 0*output
    return output


class Ensemble1(nn.Module):
    def __init__(self, models, prob_thres=0.5, size_thres=0, vote_thres=0.5, ensemble='mean'):
        super().__init__()
        self.prob_thres = prob_thres
        self.size_thres = size_thres
        self.vote_thres = vote_thres
        self.ensemble = ensemble
        self.models = models


    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            output = threshold(output, self.prob_thres)
            outputs.append(output)

        output = torch.cat(outputs)
        if self.ensemble == 'mean':
            output = torch.mean(output, dim=0)
            output = threshold(output, self.vote_thres)
            if torch.sum(output) < self.size_thres:
                output = 0*output
        return output


    @torch.no_grad()
    def predict(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            output = threshold(output, self.prob_thres)
            outputs.append(output)

        output = torch.cat(outputs)
        if self.ensemble == 'mean':
            output = torch.mean(output, dim=0)
            output = threshold(output, self.vote_thres)
            if torch.sum(output) < self.size_thres:
                output = 0*output
        return output