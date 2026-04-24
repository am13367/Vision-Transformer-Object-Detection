import argparse
import os
import random
import time
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.ops import box_iou
from transformers import DetrImageProcessor
import torch.optim as optim
from PIL import Image  # Ensure this is imported

# Import our modular files
from dataset import MatchedPairDataset
from model import build_model

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser(description="Train DETR on pixel-diff images (Option 2)")
    p.add_argument("--base_dir", type=str, required=True, help="Path to cv_data_hw2")
    p.add_argument("--matches_dir", type=str, required=True, help="Path to matched_annotations")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--freeze_mode", default="all")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# --- COLLATE FN (THE FIX) ---
def get_collate_fn(processor):
    def collate(batch):
        images, targets = zip(*batch)
        images_f, targets_hf, orig_targets = [], [], []

        for img, t in zip(images, targets):
            # Capture ORIGINAL size (w, h) before transforms
            w_orig, h_orig = img.size 
            
            # Inject size into target for evaluation later
            t["orig_size"] = torch.tensor([h_orig, w_orig]) 

            ann_list = []
            for box, label in zip(t["boxes"], t["labels"]):
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                if w <= 1 or h <= 1: continue

                ann_list.append({
                    "image_id": int(t["image_id"].item()),
                    "bbox": [x1, y1, w, h],
                    "category_id": int(label.item()),
                    "area": float(w * h),
                    "iscrowd": 0,
                })

            images_f.append(img)
            targets_hf.append({
                "image_id": int(t["image_id"].item()),
                "annotations": ann_list
            })
            orig_targets.append(t)

        if not images_f: return None

        enc = processor(images=images_f, annotations=targets_hf, return_tensors="pt")
        enc["orig_targets"] = orig_targets 
        return enc
    return collate

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    batches = 0

    for batch in loader:
        if batch is None: continue

        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)

        labels = [{
            "class_labels": t["class_labels"].to(device),
            "boxes": t["boxes"].to(device)
        } for t in batch["labels"]]

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / max(1, batches)

def eval_one_epoch(model, loader, device, processor, score_thresh=0.1, iou_thresh=0.5):
    # NOTE: Default score_thresh lowered to 0.1 to catch more boxes
    model.eval()
    total_loss = 0.0
    batches = 0
    tp = fp = fn = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None: continue

            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)

            labels = [{
                "class_labels": t["class_labels"].to(device),
                "boxes": t["boxes"].to(device)
            } for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            total_loss += outputs.loss.item()
            batches += 1

            # --- CRITICAL FIX: USE REAL SIZES ---
            # We extract the sizes we saved in collate_fn
            target_sizes = torch.stack([t["orig_size"] for t in batch["orig_targets"]]).to(device)

            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0
            )

            for pred, tgt in zip(results, batch["orig_targets"]):
                keep = pred["scores"] >= score_thresh
                pred_boxes = pred["boxes"][keep].to(device)
                pred_labels = pred["labels"][keep].to(device)
                
                gt_boxes = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)

                if len(gt_boxes) == 0:
                    fp += len(pred_boxes)
                    continue
                
                if len(pred_boxes) == 0:
                    fn += len(gt_boxes)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                matched_gt = set()

                for i in range(len(pred_boxes)):
                    if ious.shape[1] == 0: continue 
                    
                    # Friend's cleaner logic for matching
                    best_iou_val, best_iou_idx = ious[i].max(dim=0)
                    best_iou_idx = best_iou_idx.item()

                    if best_iou_idx in matched_gt:
                        fp += 1 # Already matched
                        continue

                    if best_iou_val >= iou_thresh and pred_labels[i] == gt_labels[best_iou_idx]:
                        tp += 1
                        matched_gt.add(best_iou_idx)
                    else:
                        fp += 1
                
                fn += len(gt_boxes) - len(matched_gt)

    avg_loss = total_loss / max(1, batches)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    return avg_loss, precision, recall, tp, fp, fn

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    print("Loading dataset...")
    full_ds = MatchedPairDataset(args.base_dir, args.matches_dir)
    
    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    print(f"Train: {train_len}, Val: {val_len}")

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    collate = get_collate_fn(processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    model = build_model(num_classes=6, freeze_mode=args.freeze_mode).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, prec, rec, tp, fp, fn = eval_one_epoch(model, val_loader, device, processor)
        
        duration = time.time() - start_time
        
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Prec: {prec:.3f} | Rec: {rec:.3f} | "
              f"TP/FP/FN: {tp}/{fp}/{fn} | Time: {duration:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save dict compatible with model.py loading
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"  --> Saved Best Model (Loss: {val_loss:.4f})")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()