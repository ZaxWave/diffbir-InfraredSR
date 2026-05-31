import os
import random
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .utils import augment


class SatVideoIRSDTDataset(Dataset):
    """
    红外遥感视频超分专用 Dataset。

    数据目录结构:
        {data_root}/{split}/GT/videoirstd_{orig_split}_{frame_id}[_subframe].png
        {data_root}/{split}/LQ/videoirstd_{orig_split}_{frame_id}[_subframe].png

    索引文件 {split}.txt: 每行为 "{orig_split}\\t{frame_id}"。

    在线随机滑窗裁剪 (Online Random Patch Crop):
        - GT 256×256 → 直接返回
        - GT 1024×1024 或 640×512 → 随机 crop 256×256 patch
        - LQ 坐标按 scale = gt_h // lq_h 等比例对齐
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        patch_size: int = 256,
        use_augment: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.patch_size = patch_size
        self.use_augment = use_augment

        # 读取索引
        index_path = os.path.join(data_root, f"{split}.txt")
        with open(index_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.samples: list[Dict[str, str]] = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) == 2:
                orig_split, frame_id = parts
            else:
                # 容错：如果分隔符不是 tab
                parts = line.split()
                orig_split, frame_id = parts[0], parts[1]

            gt_path = os.path.join(
                data_root, split, "GT", f"videoirstd_{orig_split}_{frame_id}.png"
            )
            lq_path = os.path.join(
                data_root, split, "LQ", f"videoirstd_{orig_split}_{frame_id}.png"
            )
            self.samples.append({"gt_path": gt_path, "lq_path": lq_path})

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        """加载图像并返回 [0, 1] float32 的 np.ndarray, shape (H, W, C), RGB."""
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0

    def _random_crop_pair(
        self, gt: np.ndarray, lq: np.ndarray, patch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GT 和 LQ 同步随机 crop，确保空间对齐。

        Args:
            gt: (H_gt, W_gt, 3), [0, 1]
            lq: (H_lq, W_lq, 3), [0, 1]
            patch_size: GT 侧的 crop 尺寸

        Returns:
            (gt_crop, lq_crop): 均为 (patch_size, patch_size, 3)
        """
        h_gt, w_gt = gt.shape[:2]
        h_lq, w_lq = lq.shape[:2]

        # 如果 GT 已经是 patch_size，直接返回（LQ 可能不同尺寸，需要处理）
        if h_gt == patch_size and w_gt == patch_size:
            # 对 LQ 做 resize 到 patch_size
            t = torch.from_numpy(lq.copy()).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(patch_size, patch_size), mode="bicubic", antialias=True)
            lq_resized = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
            return gt, lq_resized

        # 保证 GT 不小于 patch_size
        assert h_gt >= patch_size and w_gt >= patch_size, (
            f"GT ({h_gt}×{w_gt}) smaller than patch_size ({patch_size})"
        )

        # 随机生成 GT 侧的 crop 起始坐标
        top = random.randint(0, h_gt - patch_size)
        left = random.randint(0, w_gt - patch_size)

        # 等比例映射到 LQ 侧
        scale_h = h_lq / h_gt
        scale_w = w_lq / w_gt

        lq_top = int(round(top * scale_h))
        lq_left = int(round(left * scale_w))
        lq_patch_h = max(1, int(round(patch_size * scale_h)))
        lq_patch_w = max(1, int(round(patch_size * scale_w)))

        # 确保 LQ 不越界
        lq_top = min(lq_top, h_lq - lq_patch_h)
        lq_left = min(lq_left, w_lq - lq_patch_w)

        # 裁剪
        gt_crop = gt[top : top + patch_size, left : left + patch_size, :]
        lq_crop = lq[lq_top : lq_top + lq_patch_h, lq_left : lq_left + lq_patch_w, :]

        # 将 LQ crop resize 到 patch_size
        t = torch.from_numpy(lq_crop.copy()).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(patch_size, patch_size), mode="bicubic", antialias=True)
        lq_crop = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()

        return gt_crop, lq_crop

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        # 加载 GT 和 LQ
        gt = self._load_image(sample["gt_path"])  # (H, W, 3), [0,1], RGB
        lq = self._load_image(sample["lq_path"])  # (h, w, 3), [0,1], RGB

        # 同步随机 crop
        gt, lq = self._random_crop_pair(gt, lq, self.patch_size)

        # 数据增广（翻转/旋转，GT 和 LQ 同步）
        if self.use_augment and self.split == "train":
            # augment 要求 [0,1] BGR, 这里我们是 RGB, 转一下再转回
            gt_bgr = gt[..., ::-1].copy()
            lq_bgr = lq[..., ::-1].copy()
            [gt_bgr, lq_bgr] = augment([gt_bgr, lq_bgr])
            gt = gt_bgr[..., ::-1].copy()
            lq = lq_bgr[..., ::-1].copy()

        # 转换到 [-1, 1]（与 DiffBIR 的 CodeformerDataset 保持一致）
        gt = gt * 2 - 1
        # LQ 保持 [0, 1]（与 train_stage2.py 中 lq 的处理一致）

        # 转为 tensor: (H, W, C) 格式，DataLoader 堆叠后为 (B, H, W, C)
        # 与 train_stage2.py 中 rearrange('b h w c -> b c h w') 匹配
        gt = torch.from_numpy(gt.copy()).float()
        lq = torch.from_numpy(lq.copy()).float()

        return gt, lq, ""
