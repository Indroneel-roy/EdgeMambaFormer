import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)



def color_exchange(img, ref):
    """SANet Algorithm 1: transfer the LAB channel mean/std of `ref` onto `img`.

    img, ref: uint8 RGB arrays. Returns uint8 RGB with img's content, ref's colour.
    """
    a = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    b = cv2.cvtColor(ref.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    m1, s1 = a.mean((0, 1)), a.std((0, 1)) + 1e-6
    m2, s2 = b.mean((0, 1)), b.std((0, 1))
    a = (a - m1) / s1 * s2 + m2                                   # Alg. 1, line 3
    out = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_LAB2RGB)   # Alg. 1, line 5
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def elastic_3x3(img, msk, rng, sigma=10.0):
    """U-Net Sec. 3.1: random displacement vectors on a coarse 3x3 grid, sampled
    from a Gaussian with `sigma` px std; per-pixel displacements by bicubic
    interpolation."""
    h, w = msk.shape
    dx = cv2.resize(rng.normal(0, sigma, (3, 3)).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(rng.normal(0, sigma, (3, 3)).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = xs + dx, ys + dy
    img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    msk = cv2.remap(msk, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return img, msk


def shift_rotate(img, msk, rng, shift=0.1, rot=15.0):
    """Shift and rotation invariance (U-Net Sec. 3.1). Magnitudes are not given
    in the paper."""
    h, w = msk.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-rot, rot), 1.0)
    M[:, 2] += (rng.uniform(-shift, shift) * w, rng.uniform(-shift, shift) * h)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return img, msk


def flips(img, msk, rng):
    if rng.random() < 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.5: img, msk = img[::-1], msk[::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(msk)


def rot90(img, msk, rng):
    k = int(rng.integers(0, 4))
    return np.ascontiguousarray(np.rot90(img, k)), np.ascontiguousarray(np.rot90(msk, k))


def aug_unet(img, msk, rng):
    """U-Net Sec. 3.1: shift, rotation, elastic deformation, grey-value variation."""
    img, msk = flips(img, msk, rng)
    img, msk = shift_rotate(img, msk, rng)
    img, msk = elastic_3x3(img, msk, rng)
    a, b = rng.uniform(0.9, 1.1), rng.uniform(-10, 10)      # grey-value variation
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return img, msk


def aug_flip(img, msk, rng):
    return flips(img, msk, rng)


def aug_sanet(img, msk, rng):
    """SANet Sec. 4.1: random flip and random rotation."""
    img, msk = flips(img, msk, rng)
    return rot90(img, msk, rng)


def aug_none(img, msk, rng):
    return img, msk


def pcs_numpy(logit):
    """SANet Eq. 3-4 on one image: positive logits / rate_p, negative / rate_n."""
    pos, neg = logit > 0, logit < 0
    rp, rn = pos.mean(), neg.mean()
    out = logit.astype(np.float64).copy()
    if rp > 0: out[pos] = logit[pos] / rp
    if rn > 0: out[neg] = logit[neg] / rn
    return out


AUGMENTATIONS = {'unet': aug_unet, 'unetpp': aug_flip, 'pranet': aug_none, 'sanet': aug_sanet}


class BaselineTrainDataset(Dataset):
    """Resize to `size`, optional colour exchange, then a method-specific augmentation."""

    def __init__(self, pairs, aug, size=352, color_exchange_on=False):
        self.p, self.aug, self.size, self.ce = pairs, aug, size, color_exchange_on

    def __len__(self):
        return len(self.p)

    def _load(self, i):
        ip, mp = self.p[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        return (cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR),
                cv2.resize(msk, (self.size, self.size), interpolation=cv2.INTER_NEAREST))

    def __getitem__(self, i):
        rng = np.random.default_rng(np.random.randint(2 ** 31))
        img, msk = self._load(i)
        if self.ce:                                           # SANet Alg. 1
            ref, _ = self._load(int(rng.integers(len(self.p))))
            img = color_exchange(img, ref)
        img, msk = self.aug(img, msk, rng)
        x = torch.from_numpy(((img.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
        y = torch.from_numpy((msk > 127).astype(np.float32))[None]
        return x, y


class PILTrainDataset(Dataset):
    """Polyp-PVT's pipeline: PIL resize, ToTensor, ImageNet normalisation, no augmentation."""

    def __init__(self, pairs, size=352):
        from PIL import Image
        import torchvision.transforms as T
        self.Image = Image
        self.p = pairs
        self.itf = T.Compose([T.Resize((size, size)), T.ToTensor(),
                              T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gtf = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        ip, mp = self.p[i]
        return (self.itf(self.Image.open(ip).convert('RGB')),
                self.gtf(self.Image.open(mp).convert('L')))
