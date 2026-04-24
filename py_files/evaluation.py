import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput

# Import from our modular files
from dataset import MatchedPairDataset
from model import build_model, load_processor

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DETR on pixel-diff images")
    p.add_argument("--base_dir", required=True, help="Path to cv_data_hw2")
    p.add_argument("--matches_dir", required=True, help="Path to matched_annotations")
    p.add_argument("--checkpoint", required=True, help="Path to model.pth or best_model.pth")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--freeze_mode", default="all")
    p.add_argument("--score_thresh", type=float, default=0.3)
    p.add_argument("--iou_thresh", type=float, default=0.5)
    return p.parse_args()

# --- COLLATE FN (Updated to match train.py) ---
def collate_with_processor(processor):
    def _collate(batch):
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
                w, h = (x2 - x1), (y2 - y1)
                
                # Filter noise
                if w <= 1 or h <= 1: continue
                
                ann_list.append({
                    "image_id": int(t["image_id"].item()),
                    "bbox": [x1, y1, w, h],
                    "category_id": int(label.item()),
                    "area": float(w * h),
                    "iscrowd": 0,
                })

            # Always include image even if empty
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
    return _collate

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    processor = load_processor()
    ds = MatchedPairDataset(args.base_dir, args.matches_dir)
    print(f"Evaluator found {len(ds)} samples.")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_with_processor(processor)
    )

    # 2. Load Model
    model = build_model(num_classes=6, freeze_mode=args.freeze_mode).to(device)
    
    # Load Checkpoint safely
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle if checkpoint is full dict or just state_dict
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 3. Evaluation Loop
    tp = fp = fn = 0
    batches = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue

            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)

            # Forward Pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            batches += 1

            # Post-Process - UPDATED FIX
            # Use 'orig_size' which we injected in the collate function
            target_sizes = torch.stack([t["orig_size"] for t in batch["orig_targets"]]).to(device)
            
            # Use HuggingFace Output wrapper
            hf_out = DetrObjectDetectionOutput(logits=outputs.logits, pred_boxes=outputs.pred_boxes)
            results = processor.post_process_object_detection(hf_out, target_sizes=target_sizes, threshold=0.0)

            # IoU Matching Logic
            for pred, tgt in zip(results, batch["orig_targets"]):
                
                # Filter by Score Threshold
                keep = pred["scores"] >= args.score_thresh
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

                    best_iou, best_idx = ious[i].max(dim=0)
                    best_idx = best_idx.item()

                    if best_idx in matched_gt:
                        fp += 1 # Duplicate detection
                        continue

                    if best_iou >= args.iou_thresh and pred_labels[i] == gt_labels[best_idx]:
                        tp += 1
                        matched_gt.add(best_idx)
                    else:
                        fp += 1
                
                fn += len(gt_boxes) - len(matched_gt)

    # 4. Summary
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-6, precision + recall)

    print("\n" + "="*40)
    print(f"EVAL RESULTS (Score Thresh: {args.score_thresh})")
    print("="*40)
    print(f"True Positives  : {tp}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print("-" * 20)
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()