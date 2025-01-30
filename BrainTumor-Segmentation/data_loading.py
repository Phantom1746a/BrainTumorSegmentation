import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def get_data_paths(root_dir):
    patients = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    data_paths = []
    for patient in patients:
        patient_dir = os.path.join(root_dir, patient)
        modalities = [
            os.path.join(patient_dir, f'{patient}_t1.nii.gz'),
            os.path.join(patient_dir, f'{patient}_t1ce.nii.gz'),
            os.path.join(patient_dir, f'{patient}_t2.nii.gz'),
            os.path.join(patient_dir, f'{patient}_flair.nii.gz'),
        ]
        mask_path = os.path.join(patient_dir, f'{patient}_seg.nii.gz')
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        depth = mask_data.shape[2]
        for slice_idx in range(depth):
            data_paths.append((modalities, mask_path, slice_idx))
    return data_paths

class BraTSDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        modalities_paths, mask_path, slice_idx = self.data_paths[idx]
        
        # Load modalities
        modalities = []
        for mod_path in modalities_paths:
            img = nib.load(mod_path).get_fdata()
            img = self.normalize(img[..., slice_idx])
            modalities.append(img)
        image = np.stack(modalities, axis=-1)  # (H, W, 4)
        
        # Load mask
        mask = nib.load(mask_path).get_fdata()[..., slice_idx]
        wt = np.logical_or(mask == 1, np.logical_or(mask == 2, mask == 4))
        tc = np.logical_or(mask == 1, mask == 4)
        et = mask == 1
        mask = np.stack([wt, tc, et], axis=-1).astype(np.float32)  # (H, W, 3)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (4, H, W)
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()    # (3, H, W)
        
        if self.transform:
            combined = torch.cat([image, mask], dim=0)
            combined = self.transform(combined)
            image = combined[:4]
            mask = combined[4:]
            
        return image, mask

    def normalize(self, img):
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        img = (img - p1) / (p99 - p1 + 1e-6)
        return img