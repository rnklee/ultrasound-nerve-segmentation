class NerveTestset(Dataset):
    def __init__(self, imgfiles, preprocess=None):
        super().__init__()

        self.imgfiles = [testpath+o for o in imgfiles]
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


@torch.no_grad()
def predict_test(test_imgs, model, preprocess=None, postprocess=None, device='cuda'):
    if isinstance(model, str):
        test_model = torch.load(model)
    else:
        test_model = model

    model.eval()

    test_data = NerveTestset(imgfiles=test_imgs, preprocess=preprocess)
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
            pred = test_model(x).round().cpu().detach().numpy()
            pred = pred.squeeze()

            if postprocess is not None:
                pred = default_postprocess(pred)

            test_preds.append(pred)

    return test_preds


def create_submission(test_imgs, model, preprocess, postprocess=None, filename=None):
    test_preds = predict_test(test_imgs, model, preprocess, postprocess)

    if filename is None:
        filename = 'submission' + datetime.now().strftime('_%m%d%H%M') +'.csv'

    first_row = 'img,pixels'
    with open(filename, 'w+') as f:
        f.write(first_row + '\n')
        for i, pred in enumerate(test_preds):
            s = str(test_imgs[i].replace('.tif', '')) + ',' + rle_encoding(pred)
            f.write(s + '\n')

    files.download(filename)

    return test_preds