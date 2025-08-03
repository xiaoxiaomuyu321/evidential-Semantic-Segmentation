import os
import argparse
import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler, autocast
import pandas as pd
from tqdm import tqdm  # ✅ 导入 tqdm
from time import time
from evidence_semantic_segmentation.data_loader.custom_data import create_dataloader
from evidence_semantic_segmentation.Model import Segmentation
from evidence_semantic_segmentation.utils.metrics import label_accuracy_score
from evidence_semantic_segmentation.losses import SegLoss
from thop import profile
from thop import clever_format

# -------------------- 单轮训练 ------------------
def train_one_epoch(model, loader, criterion, optimizer, device, num_classes, scaler, use_amp, epoch, epoches):
    model.train()
    total_loss = 0.0
    all_true, all_pred = [], []
    start_time = time()

    pbar = tqdm(loader, desc=f"🚀 train Epoch {epoch}/{epoches}", leave=True, ncols=100)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            outputs = F.log_softmax(outputs, dim=1)
            loss = criterion(outputs, labels, epoch)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        all_true.append(labels.detach().cpu())
        all_pred.append(preds.detach().cpu())

        elapsed = time() - start_time

        # 更新进度条内容
        pbar.set_postfix({'loss': f"{loss.item():.4f}",
                          'time': f"{elapsed:.1f}s"})

    avg_loss = total_loss / len(loader.dataset)
    true_np = torch.cat(all_true, dim=0).numpy()
    pred_np = torch.cat(all_pred, dim=0).numpy()
    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(true_np, pred_np, num_classes)
    pbar.set_postfix({'avg_loss': f"{avg_loss:.4f}", 'time': f"{time() - start_time:.1f}s"})  # ✅ 核心更改
    return avg_loss, acc, acc_cls, mean_iu, fwavacc

# -------------------- 单轮验证 ------------------
def validate_one_epoch(model, loader, criterion, device, num_classes, use_amp, epoch, epoches):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []
    start_time = time()

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"🔍 Val   Epoch {epoch}/{epoches}", leave=True, ncols=100)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                outputs = F.log_softmax(outputs, dim=1)
                loss = criterion(outputs, labels, epoch)

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            all_true.append(labels.detach().cpu())
            all_pred.append(preds.detach().cpu())

            elapsed = time() - start_time


            # 更新进度条内容
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'time': f"{elapsed:.1f}s"
            })

    avg_loss = total_loss / len(loader.dataset)
    true_np = torch.cat(all_true, dim=0).numpy()
    pred_np = torch.cat(all_pred, dim=0).numpy()
    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(true_np, pred_np, num_classes)
    if epoch ==epoches:
        tqdm.write(f"✅ Final val | Loss: {avg_loss:.4f} | acc: {acc:.4f} | acc_cls: {acc_cls:.4f} | mIoU: {mean_iu:.4f} | fwavacc: {fwavacc:.4f}")

    return avg_loss, acc, acc_cls, mean_iu, fwavacc

def save_csv_log(log_path, record_dict):
    df = pd.DataFrame([record_dict])
    if not os.path.exists(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode='a', header=False, index=False)


def train(model, train_loader, val_loader, criterion, optimizer, device, args):
    best_score = 0.0
    scaler = GradScaler(device='cuda', enabled=args.amp)

    base_dir = os.path.join(args.project_name, args.model_name)
    os.makedirs(base_dir, exist_ok=True)
    train_log_path = os.path.join(base_dir, 'train.csv')
    val_log_path = os.path.join(base_dir, 'val.csv')

    for epoch in range(1, args.epochs + 1):
        train_loss, acc, acc_cls, mean_iu, fwavacc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.num_classes, scaler, args.amp, epoch, args.epochs)

        save_csv_log(train_log_path, {
            'epoch': epoch,
            'loss': train_loss,
            'acc': acc,
            'acc_cls': acc_cls,
            'miou': mean_iu,
            'fwavacc': fwavacc
        })

        # 验证条件
        if epoch >= 280 and epoch % 5 == 0 or epoch % 2 == 0:
            val_loss, val_acc, val_acc_cls, val_mean_iu, val_fwavacc = validate_one_epoch(
                model, val_loader, criterion, device, args.num_classes, args.amp, epoch, args.epochs)

            tqdm.write(f"✅ Epoch {epoch} | Val Loss: {val_loss:.4f} | acc: {val_acc:.4f} | acc_cls: {val_acc_cls:.4f} | mIoU: {val_mean_iu:.4f} | fwavacc: {val_fwavacc:.4f}")

            score = (val_acc_cls + val_mean_iu) / 2
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
                tqdm.write(f"✅ Best model saved at epoch {epoch:03d} with score {score:.4f}")

            save_csv_log(val_log_path, {
                'epoch': epoch,
                'loss': val_loss,
                'acc': val_acc,
                'acc_cls': val_acc_cls,
                'miou': val_mean_iu,
                'fwavacc': val_fwavacc
            })

# -------------------- 主函数入口 ------------------
def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Training with AMP & Logging")

    # 参数定义
    parser.add_argument('--data_root', type=str, default='datasets/VOCdevkit/VOC2012')
    parser.add_argument('--project_name', type=str, default='voc_project')
    parser.add_argument('--model_name', type=str, default='UNet')
    parser.add_argument('--input_width', type=int, default=320)
    parser.add_argument('--input_height', type=int, default=320)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--amp', type=bool, default=True, help='Enable AMP (mixed precision)')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Segmentation(model_name=args.model_name, num_classes=args.num_classes).to(device)

    # 创建一个虚拟输入张量（模拟一个 batch）
    dummy_input = torch.randn(1, 3, args.input_height, args.input_width).to(device)
    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], "%.3f")
    tqdm.write(f"🧠 Total Parameters: {params} ⚙️ Total FLOPs: {flops}")

    train_loader = create_dataloader(args.data_root, 'train', args.input_width, args.input_height,
                                     args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = create_dataloader(args.data_root, 'val', args.input_width, args.input_height,
                                   args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    criterion = SegLoss(num_classes=args.num_classes,annealing_step=100, loss_type="edl_mse").to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, val_loader, criterion, optimizer, device, args)

if __name__ == "__main__":
    main()
