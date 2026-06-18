import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import Config
from src.dataset import build_dataloaders
import src.dataset as sd

class DatasetInfo:
    def __init__(self, val_size, image_size):
        self.name = "COCO Person"
        self.num_classes = 1
        self.type = "binary"
        self.class_names = ["person"]
        self.image_size = image_size
        self.val_size = val_size
        self.palette = np.array([[0, 0, 0], [255, 0, 0]]) # Background (black), Person (red)
        self.ignore_index = 255 # standard ignore label for dataset

class SyntheticCOCODataset(Dataset):
    """A synthetic dataset that mimics COCO dataset structure for testing/fallback when COCO is not present."""
    def __init__(self, size=10, image_size=(320, 320)):
        self.size = size
        self.image_size = image_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate a synthetic image (normalized with typical ImageNet stats)
        img = torch.randn(3, *self.image_size)
        # Create a simple synthetic circle mask representing a person
        mask = torch.zeros(1, *self.image_size)
        h, w = self.image_size
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        dist = (y - cy)**2 + (x - cx)**2
        mask[0, dist < r**2] = 1.0
        return img, mask

def create_dataloaders(dataset_name, image_size=(320, 320), batch_size=1, num_workers=4, aug_intensity="light"):
    """
    Creates dataloaders. If real COCO dataset files are missing, it falls back to a synthetic dataset
    so that notebooks can run seamlessly.
    """
    cfg = Config(
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0 if num_workers == 0 else 2, # reduce workers for stability
        prefetch_factor=2
    )
    
    # Check if COCO directory or annotations exist
    coco_exists = (
        cfg.train_ann_file is not None and 
        cfg.val_ann_file is not None and 
        cfg.train_ann_file.endswith(".json") and 
        cfg.val_ann_file.endswith(".json") and 
        cfg.train_ann_file != "" and 
        cfg.val_ann_file != "" and
        __import__("os").path.exists(cfg.train_ann_file) and
        __import__("os").path.exists(cfg.val_ann_file)
    )
    
    if coco_exists:
        try:
            train_loader, val_loader = build_dataloaders(cfg)
            info = DatasetInfo(len(val_loader.dataset), image_size)
            return train_loader, val_loader, info
        except Exception as e:
            print(f"Failed to load real COCO dataset: {e}. Falling back to synthetic dataset.")
            
    # Fallback to synthetic
    print("⚠️ COCO dataset annotations not found or failed to load. Using Synthetic dataset fallback.")
    train_dataset = SyntheticCOCODataset(size=20, image_size=image_size)
    val_dataset = SyntheticCOCODataset(size=10, image_size=image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    info = DatasetInfo(len(val_dataset), image_size)
    return train_loader, val_loader, info

def denormalize(image_tensor):
    """Wrapper that returns HWC numpy array, ready for matplotlib imshow."""
    res = sd.denormalize(image_tensor.cpu())
    if isinstance(res, torch.Tensor):
        res = res.permute(1, 2, 0).numpy()
    return res

def label_to_color(mask, palette):
    """Converts a binary/labeled mask to an RGB image using the specified palette."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[mask > 0.5] = palette[1]
    colored[mask <= 0.5] = palette[0]
    return colored
