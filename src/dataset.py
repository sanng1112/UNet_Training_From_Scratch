"""Dataset & DataLoader cho Human Segmentation trên COCO 2017.

Chỉ giữ lại ảnh có chứa nhãn 'person'. Mask nhị phân hoá từ tất cả annotation
person trong ảnh (hợp các instance). Augmentation huấn luyện gồm: resize cạnh
ngắn, RandomResizedCrop, horizontal flip, Copy-Paste, ColorJitter, GaussianBlur.
"""
from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# Thống kê ImageNet để chuẩn hoá ảnh đầu vào.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PersonStampPool:
    """Pool các person stamp đã cắt sẵn để Copy-Paste nhanh hơn.

    Thay vì load ảnh B mỗi lần Copy-Paste (gây I/O ngốn CPU), pool này
    precompute N person stamp từ random ảnh và random pick từ pool.
    """

    def __init__(
        self,
        coco: COCO,
        img_dir: str,
        img_ids: list,
        pool_size: int = 500,
        min_person_ratio: float = 0.005,
    ) -> None:
        self.coco = coco
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.pool_size = pool_size
        self.min_person_ratio = min_person_ratio
        self.pool: list = []
        self._build_pool()

    def _build_pool(self) -> None:
        """Xây dựng pool các person stamp từ random ảnh."""
        n = min(self.pool_size, len(self.img_ids))
        sampled_ids = random.sample(self.img_ids, n)
        person_cat_id = self.coco.getCatIds(catNms=["person"])[0]
        for img_id in sampled_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info["file_name"])
            try:
                image = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
            h, w = image.shape[:2]

            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                mask = self.coco.annToMask(ann)
                person_pixels = mask.sum()
                if person_pixels < self.min_person_ratio * h * w:
                    continue  # bỏ stamp quá nhỏ
                ys, xs = np.where(mask)
                y1, y2 = ys.min(), ys.max() + 1
                x1, x2 = xs.min(), xs.max() + 1
                person_crop = image[y1:y2, x1:x2].copy()
                person_mask = mask[y1:y2, x1:x2].copy()
                self.pool.append((person_crop, person_mask))
        if not self.pool:
            print("⚠️  PersonStampPool trống! Copy-Paste sẽ dùng fallback.")

    def sample(self):
        """Lấy ngẫu nhiên một (person_crop, person_mask) từ pool."""
        return random.choice(self.pool)
class COCOPersonDataset(Dataset):
    """Dataset phân vùng người. Toàn bộ phép toán không gian xử lý trên NumPy/PIL
    để giảm ép kiểu, các phép Copy-Paste làm trực tiếp trên mảng nhị phân."""

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        image_size: Tuple[int, int] = (320, 320),
        is_train: bool = True,
        copy_paste_prob: float = 0.5,
    ) -> None:
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.image_size = image_size
        self.is_train = is_train
        self.copy_paste_prob = copy_paste_prob

        self.person_cat_id = self.coco.getCatIds(catNms=["person"])[0]
        self.img_ids = self.coco.getImgIds(catIds=[self.person_cat_id])

    def __len__(self) -> int:
        return len(self.img_ids)

    # ------------------------------------------------------------------ #
    def _load_raw_numpy(self, idx: int):
        """Đọc ảnh RGB + mask person (uint8) dưới dạng mảng NumPy."""
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.person_cat_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))
        return image, mask

    def _spatial_aug(self, image: np.ndarray, mask: np.ndarray):
        """Resize cạnh ngắn -> RandomResizedCrop -> HorizontalFlip (cho train)."""
        img_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask * 255)

        short_edge = min(self.image_size)
        img_pil = TF.resize(img_pil, short_edge, interpolation=InterpolationMode.BILINEAR)
        mask_pil = TF.resize(mask_pil, short_edge, interpolation=InterpolationMode.NEAREST)

        i, j, h, w = T.RandomResizedCrop.get_params(img_pil, scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
        img_pil = TF.resized_crop(img_pil, i, j, h, w, size=self.image_size, interpolation=InterpolationMode.BILINEAR)
        mask_pil = TF.resized_crop(mask_pil, i, j, h, w, size=self.image_size, interpolation=InterpolationMode.NEAREST)

        if random.random() > 0.5:
            img_pil = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)

        return np.array(img_pil), np.array(mask_pil) > 0

    def _val_preprocess(self, image: np.ndarray, mask: np.ndarray):
        """Resize cạnh ngắn + center crop (xác định, không ngẫu nhiên)."""
        img_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask * 255)
        short_edge = min(self.image_size)
        img_pil = TF.resize(img_pil, short_edge, interpolation=InterpolationMode.BILINEAR)
        mask_pil = TF.resize(mask_pil, short_edge, interpolation=InterpolationMode.NEAREST)
        img_pil = TF.center_crop(img_pil, self.image_size)
        mask_pil = TF.center_crop(mask_pil, self.image_size)
        return img_pil, np.array(mask_pil) > 0

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int):
        image, mask = self._load_raw_numpy(idx)

        if self.is_train:
            image, mask = self._spatial_aug(image, mask)

            # Copy-Paste: phủ đối tượng người từ ảnh B lên ảnh A.
            if random.random() < self.copy_paste_prob:
                if hasattr(self, 'copy_paste_pool') and self.copy_paste_pool is not None \
                        and len(self.copy_paste_pool.pool) > 0:
                    # Dùng pool: nhanh hơn vì không load ảnh mới
                    person_crop, person_mask = self.copy_paste_pool.sample()
                    crop_h, crop_w = person_crop.shape[:2]
                    # Resize crop về kích thước ngẫu nhiên (0.3~1.0 lần)
                    scale = random.uniform(0.3, 1.0)
                    new_h, new_w = max(1, int(crop_h * scale)), max(1, int(crop_w * scale))
                    person_crop = np.array(Image.fromarray(person_crop).resize(
                        (new_w, new_h), Image.BILINEAR))
                    person_mask = np.array(Image.fromarray(
                        (person_mask * 255).astype(np.uint8)).resize(
                        (new_w, new_h), Image.NEAREST)) > 0
                    # Paste vào vị trí ngẫu nhiên
                    img_h, img_w = image.shape[:2]
                    paste_y = random.randint(0, max(0, img_h - new_h))
                    paste_x = random.randint(0, max(0, img_w - new_w))
                    actual_h = min(new_h, img_h - paste_y)
                    actual_w = min(new_w, img_w - paste_x)
                    crop_region = (slice(paste_y, paste_y + actual_h),
                                   slice(paste_x, paste_x + actual_w))
                    person_region = person_mask[:actual_h, :actual_w]
                    image[crop_region][person_region] = person_crop[:actual_h, :actual_w][person_region]
                    mask_crop = mask[crop_region]
                    mask_crop[person_region] = 1
                    mask[crop_region] = mask_crop
                else:
                    # Fallback: load ảnh B (cách cũ)
                    idx_b = random.randint(0, len(self.img_ids) - 1)
                    image_b, mask_b = self._load_raw_numpy(idx_b)
                    image_b, mask_b = self._spatial_aug(image_b, mask_b)
                    image[mask_b] = image_b[mask_b]
                    mask = np.maximum(mask, mask_b)

            img_pil = Image.fromarray(image)
            if random.random() > 0.5:
                img_pil = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(img_pil)
            if random.random() > 0.7:
                img_pil = T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))(img_pil)
            image_tensor = TF.to_tensor(img_pil)
        else:
            img_pil, mask = self._val_preprocess(image, mask)
            image_tensor = TF.to_tensor(img_pil)

        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        return image_tensor, mask_tensor


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    """Đảo chuẩn hoá ImageNet để hiển thị ảnh (trả về tensor CHW trong [0, 1])."""
    mean = torch.tensor(IMAGENET_MEAN, device=image_tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_tensor.device).view(-1, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Tạo train_loader & val_loader từ Config."""
    train_ds = COCOPersonDataset(
        cfg.train_img_dir, cfg.train_ann_file,
        image_size=cfg.image_size, is_train=True, copy_paste_prob=cfg.copy_paste_prob,
    )
    val_ds = COCOPersonDataset(
        cfg.val_img_dir, cfg.val_ann_file,
        image_size=cfg.image_size, is_train=False,
    )

    # Subset cho debug nhanh
    subset = getattr(cfg, 'subset', None)
    if subset is not None:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(subset, len(train_ds))))
        val_ds = Subset(val_ds, range(min(subset, len(val_ds))))
        print(f"⚠️  Subset mode: {len(train_ds)} train / {len(val_ds)} val samples")

    common = dict(
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    return train_loader, val_loader
