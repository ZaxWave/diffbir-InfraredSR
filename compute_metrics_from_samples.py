"""
从已保存的 val_results_* 样本图中计算 PSNR/LPIPS 汇总
无需重新推理，直接读保存的图片算指标
"""
import os, sys, torch, lpips, numpy as np
from PIL import Image
from glob import glob
# from tabulate import tabulate

results_dir = sys.argv[1] if len(sys.argv) > 1 else "exps/satvideo_irsr_stage2"

lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()

rows = []
for d in sorted(os.listdir(results_dir)):
    if not d.startswith("val_results_"):
        continue
    samples_dir = os.path.join(results_dir, d, "samples")
    if not os.path.isdir(samples_dir):
        continue

    # 收集所有 pred/gt 对
    pred_files = sorted(glob(os.path.join(samples_dir, "*_pred.png")))
    gt_files = sorted(glob(os.path.join(samples_dir, "*_gt.png")))

    if len(pred_files) == 0:
        continue

    psnr_list, lpips_list = [], []
    for pf, gf in zip(pred_files, gt_files):
        pred = np.array(Image.open(pf).convert("RGB")).astype(np.float32) / 255.0
        gt   = np.array(Image.open(gf).convert("RGB")).astype(np.float32) / 255.0

        # PSNR
        mse = np.mean((pred - gt) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")
        psnr_list.append(psnr)

        # LPIPS
        pred_t = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).float() * 2 - 1
        gt_t   = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            lpips_val = lpips_model(pred_t, gt_t, normalize=True).item()
        lpips_list.append(lpips_val)

    step = d.replace("val_results_", "")
    rows.append([f"{step}.pt", np.mean(psnr_list), np.std(psnr_list),
                 np.mean(lpips_list), np.std(lpips_list), len(psnr_list)])

print("\n" + "="*80)
print("  Validation Summary — {}/val_results_* sample images".format(results_dir))
print("="*80)
header = f"{'Checkpoint':>15} {'PSNR ↑':>10} {'PSNR-std':>10} {'LPIPS ↓':>10} {'LPIPS-std':>10} {'#samples':>9}"
print(header)
print("-" * len(header))
for r in rows:
    print(f"{r[0]:>15} {r[1]:>10.4f} {r[2]:>10.4f} {r[3]:>10.6f} {r[4]:>10.6f} {r[5]:>9d}")
print()

# 找最优
best_psnr = max(rows, key=lambda r: r[1])
best_lpips = min(rows, key=lambda r: r[3])
print(f"  PSNR 最优: {best_psnr[0]}  ({best_psnr[1]:.4f})")
print(f"  LPIPS 最优: {best_lpips[0]}  ({best_lpips[3]:.6f})")
print()

# 注：样本数远小于全量验证集(139张)，仅供快速参考
if rows and rows[0][5] < 50:
    print("  [注意] 仅基于保存的样本图计算，非全量验证集结果")
