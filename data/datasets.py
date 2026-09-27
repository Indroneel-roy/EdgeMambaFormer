"""
    DATA_ROOT/
      TrainDataset/
        images/*.png          900 Kvasir-SEG + 550 CVC-ClinicDB = 1450
        masks/*.png
      TestDataset/
        Kvasir/{images,masks}              100   in-distribution
        CVC-ClinicDB/{images,masks}         62   in-distribution
        CVC-ColonDB/{images,masks}         380   zero-shot
        ETIS-LaribPolypDB/{images,masks}   196   zero-shot
        CVC-300/{images,masks}              60   zero-shot  

"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')

VAL_SPLIT_SEED = 12345
VAL_FRACTION = 0.10

TEST_SETS = {
    'Kvasir':            'in-distribution',
    'CVC-ClinicDB':      'in-distribution',
    'CVC-ColonDB':       'zero-shot',
    'ETIS-LaribPolypDB': 'zero-shot',
    'CVC-300':           'zero-shot',
}


def _glob(d: Path):
    out = []
    for pat in IMG_EXTS:
        out.extend(d.glob(pat))
    return sorted(out)


def build_pairs(root, img_dir='images', mask_dir='masks'):
    """Match each image to a mask of the same stem, trying several extensions."""
    root = Path(root)
    idir, mdir = root / img_dir, root / mask_dir
    if not idir.is_dir():
        raise FileNotFoundError(f'Missing image dir: {idir}')

    pairs = []
    for ip in _glob(idir):
        mp = None
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            cand = mdir / (ip.stem + ext)
            if cand.exists():
                mp = cand
                break
        if mp is None:
            raise FileNotFoundError(f'No mask found for {ip.name} in {mdir}')
        pairs.append((str(ip), str(mp)))
    return pairs


def load_splits(data_root, verbose=True):
    """
    Returns (train_pairs, val_pairs, test_dict) with all leak assertions enforced.
    """
    data_root = Path(data_root)
    train_all = build_pairs(data_root / 'TrainDataset')

    test = {}
    for name in TEST_SETS:
        d = data_root / 'TestDataset' / name
        if d.is_dir():
            test[name] = build_pairs(d)
        elif verbose:
            print(f'  [warn] missing test set: {name}')

    # ── validation carved from TRAIN, never from TEST
    idx = list(range(len(train_all)))
    random.Random(VAL_SPLIT_SEED).shuffle(idx)
    n_val = int(VAL_FRACTION * len(train_all))
    val_pairs = [train_all[i] for i in idx[:n_val]]
    train_pairs = [train_all[i] for i in idx[n_val:]]

    # ── GATE 1: 
    def stems(ps):
        return {Path(p).stem for p, _ in ps}

    tr, va = stems(train_pairs), stems(val_pairs)
    assert not (tr & va), f'TRAIN leaks into VAL: {len(tr & va)} images'

    for name, ps in test.items():
        te = stems(ps)
        assert not (tr & te), f'TRAIN leaks into TEST[{name}]: {len(tr & te)} images'
        assert not (va & te), f'VAL leaks into TEST[{name}]: {len(va & te)} images'

    if verbose:
        print(f'  TRAIN {len(train_pairs)} | VAL {len(val_pairs)} '
              f'(from TrainDataset, seed {VAL_SPLIT_SEED})')
        for name, ps in test.items():
            print(f'  TEST  {name:<20s} {len(ps):>4d}  ({TEST_SETS[name]})')
        print('  GATE 1 PASS: no leakage between train / val / test')

    return train_pairs, val_pairs, test


#  Datasets


class PolypTrainDataset(Dataset):
    """Train/val: image and mask both resized to img_size."""

    def __init__(self, pairs, transforms):
        self.pairs = pairs
        self.tf = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

        aug = self.tf(image=img, mask=msk)
        img, msk = aug['image'], aug['mask']
        msk = (msk > 127).float().unsqueeze(0)
        return img, msk


class PolypTestDataset(Dataset):
    def __init__(self, pairs, img_size=352):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        self.pairs = pairs
        self.tf = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)          # NATIVE, untouched
        img = self.tf(image=img)['image']
        gt = torch.from_numpy((gt > 127).astype(np.float32))
        return img, gt, Path(ip).name


def make_test_loader(pairs, img_size=352, num_workers=2):
    from torch.utils.data import DataLoader
    return DataLoader(PolypTestDataset(pairs, img_size),
                      batch_size=1, shuffle=False,      # batch_size=1 is mandatory
                      num_workers=num_workers, pin_memory=True)

#  Augmentation


def get_train_transforms(img_size=352):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=352):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

#  Reproducibility


def set_seed(seed):
    """Run seed: varies init + augmentation order ONLY. Split seed is fixed above."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)
