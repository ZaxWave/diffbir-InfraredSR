"""
推理验证集脚本 — 实验一
加载训练好的 ControlNet checkpoint，对 val 集做超分，计算 PSNR / LPIPS。

用法:
  python3 infer_val.py --ckpt exps/satvideo_irsr_stage2/checkpoints/0030000.pt \
                       --config configs/train/train_stage2.yaml \
                       --output exps/satvideo_irsr_stage2/val_results

可选:
  --batch_size 4    # 根据显存调整
  --steps 50        # 扩散步数
  --device cuda:0
"""

import os, argparse, torch, lpips, numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torch.nn import functional as F
from einops import rearrange

from diffbir.model import ControlLDM, Diffusion
from diffbir.utils.common import instantiate_from_config, to
from diffbir.sampler import SpacedSampler
from diffbir.dataset.satvideo_irsr import SatVideoIRSDTDataset


def calculate_psnr(img1, img2, crop_border=4):
    """img1, img2: [0,1] float32 numpy (H,W,C)"""
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))


def main(args):
    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)

    # --- 加载模型 ---
    print("Loading ControlLDM...")
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    cldm.load_pretrained_sd(sd)
    cldm.load_controlnet_from_unet()

    # 加载训练好的 ControlNet 权重
    print(f"Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    cldm.controlnet.load_state_dict(state, strict=True)
    cldm.eval().to(device)

    # 加载 diffusion
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)

    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)

    # --- 加载验证集/测试集 ---
    data_cfg = cfg.dataset.train.params
    dataset = SatVideoIRSDTDataset(
        data_root=data_cfg.data_root,
        split=args.split,
        patch_size=256,
        use_augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False)
    print(f"Loaded {args.split} set: {len(dataset)} images")

    # --- 推理 ---
    os.makedirs(args.output, exist_ok=True)
    psnr_list, lpips_list = [], []
    prompt = ""

    pbar = tqdm(loader, desc="Infer val")
    for batch in pbar:
        to(batch, device)
        gt, lq, _ = batch  # gt: [-1,1], lq: [0,1], both (B,H,W,C)
        gt_img = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
        lq_img = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)

        b = gt_img.size(0)

        # VAE encode GT
        with torch.no_grad():
            z_0 = cldm.vae_encode(gt_img)
            # bicubic upsample 作为 condition
            clean = F.interpolate(lq_img, size=gt_img.shape[2:], mode="bicubic", antialias=True)
            cond = cldm.prepare_condition(clean, [prompt] * b)

            # 采样
            z = sampler.sample(
                model=cldm,
                device=device,
                steps=args.steps,
                x_size=(b, *z_0.shape[1:]),
                cond=cond,
                uncond=None,
                cfg_scale=1.0,
                progress=False,
            )
            pred = cldm.vae_decode(z).float()  # [-1, 1]

        # 转成 [0,1] numpy (B, H, W, C)
        pred_np = ((pred + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        gt_np = ((gt_img + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        lq_np = lq_img.permute(0, 2, 3, 1).cpu().numpy()

        for i in range(b):
            psnr_val = calculate_psnr(pred_np[i], gt_np[i])
            psnr_list.append(psnr_val)
            with torch.no_grad():
                lpips_val = lpips_model(
                    ((pred[i : i + 1] + 1) / 2).clamp(0, 1),
                    ((gt_img[i : i + 1] + 1) / 2).clamp(0, 1),
                    normalize=True,
                ).item()
            lpips_list.append(lpips_val)

            # 保存图像
            idx = len(psnr_list) - 1
            if idx < 10 or idx % 10 == 0:  # 采样展示
                os.makedirs(f"{args.output}/samples", exist_ok=True)
                pred_img = Image.fromarray((pred_np[i] * 255).astype(np.uint8))
                gt_img_pil = Image.fromarray((gt_np[i] * 255).astype(np.uint8))
                pred_img.save(f"{args.output}/samples/{idx:04d}_pred.png")
                gt_img_pil.save(f"{args.output}/samples/{idx:04d}_gt.png")
                # 左右拼接对比图
                concat = Image.new("RGB", (pred_img.width * 2, pred_img.height))
                concat.paste(gt_img_pil, (0, 0))
                concat.paste(pred_img, (pred_img.width, 0))
                concat.save(f"{args.output}/samples/{idx:04d}_compare.png")

        pbar.set_postfix(PSNR=f"{np.mean(psnr_list):.2f}")

    # --- 结果 ---
    print(f"\n=== {args.split} results ({args.ckpt}) ===")
    print(f"  PSNR :  {np.mean(psnr_list):.4f}  (std={np.std(psnr_list):.4f})")
    print(f"  LPIPS:  {np.mean(lpips_list):.6f}  (std={np.std(lpips_list):.6f})")
    print(f"  Total samples: {len(psnr_list)}")
    print(f"  Results saved to {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to controlnet checkpoint")
    parser.add_argument("--config", type=str, default="configs/train/train_stage2.yaml")
    parser.add_argument("--output", type=str, default="exps/satvideo_irsr_stage2/val_results")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)
