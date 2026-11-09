import json, os, random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class GroupDataset(Dataset):
    """Episode dataset. Manifest maps group IDs to image/mask records."""
    def __init__(self, manifest, image_size=352, group_size=5, train=True, seed=42):
        with open(manifest, "r", encoding="utf-8") as f: self.groups = json.load(f)
        self.root = os.path.dirname(os.path.abspath(manifest)); self.ids = list(self.groups.keys())
        self.image_size, self.group_size, self.train, self.seed = image_size, group_size, train, seed

    def __len__(self): return len(self.ids)

    def _path(self, p): return p if os.path.isabs(p) else os.path.join(self.root, p)

    def _load(self, rec):
        img = Image.open(self._path(rec["image"])).convert("RGB")
        mask = Image.open(self._path(rec["mask"])).convert("L") if rec.get("mask") else None
        img = TF.resize(img, [self.image_size, self.image_size], antialias=True)
        img = TF.to_tensor(img); img = TF.normalize(img, [0.485,0.456,0.406], [0.229,0.224,0.225])
        if mask is not None:
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
            mask = (TF.to_tensor(mask) > 0.5).float()
        return img, mask

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx if self.train else idx)
        records = self.groups[self.ids[idx]]
        if len(records) >= self.group_size: records = rng.sample(records, self.group_size)
        else: records = [records[i % len(records)] for i in range(self.group_size)]
        ims, ms, names = [], [], []
        for r in records:
            im, m = self._load(r); ims.append(im); names.append(os.path.basename(r["image"]))
            if m is not None: ms.append(m)
        out = {"images": torch.stack(ims), "names": names, "group_id": self.ids[idx]}
        if ms: out["masks"] = torch.stack(ms)
        return out
