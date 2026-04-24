import argparse
import os
import random
import cv2
import torch
import numpy as np
from PIL import Image

# IMPORT FROM OUR MODULES
from dataset import MatchedPairDataset
from model import build_model, load_processor, LABELS

def draw_box(img, box, color, label=None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(12, y1 - 4)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="Visualize Predictions Side-by-Side")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--matches_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="vis_predictions")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--score_thresh", type=float, default=0.1) # Lower threshold to see more
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    ds = MatchedPairDataset(args.base_dir, args.matches_dir)
    print(f"Found {len(ds)} samples.")
    
    # 2. Load Model
    model = build_model(num_classes=6, freeze_mode="all").to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" in ckpt: model.load_state_dict(ckpt["model"])
    else: model.load_state_dict(ckpt)
    model.eval()
    
    processor = load_processor()

    # 3. Random Sample Loop
    indices = random.sample(range(len(ds)), min(args.num_samples, len(ds)))
    
    for idx in indices:
        diff_img, target = ds[idx]
        
        # Prepare Input
        enc = processor(images=[diff_img], return_tensors="pt")
        pixel_values = enc["pixel_values"].to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        
        # Post Process
        target_sizes = torch.tensor([diff_img.size[::-1]]).to(device) # (h, w)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        
        # Filter Predictions
        keep = results["scores"] >= args.score_thresh
        pred_boxes = results["boxes"][keep].cpu().numpy()
        pred_labels = results["labels"][keep].cpu().numpy()
        pred_scores = results["scores"][keep].cpu().numpy()

        # Load Original Image 2 (Target Frame)
        sample_info = ds.samples[idx]
        img_base = cv2.imread(sample_info["img2"])
        
        if img_base is None: continue

        # --- CREATE SIDE-BY-SIDE VIEW ---
        
        # Left Image: Ground Truth Only (Green)
        img_gt = img_base.copy()
        gt_boxes = target["boxes"].numpy() 
        for box in gt_boxes:
            draw_box(img_gt, box, (0, 255, 0), "GT")
            
        # Right Image: Predictions Only (Red)
        img_pred = img_base.copy()
        for i, box in enumerate(pred_boxes):
            lbl = f"{LABELS.get(pred_labels[i], 'obj')} {pred_scores[i]:.2f}"
            draw_box(img_pred, box, (0, 0, 255), lbl)

        # Stitch them together (Horizontal Concatenation)
        combined_img = cv2.hconcat([img_gt, img_pred])

        # Save
        match_id = os.path.basename(sample_info["ann"])[:-10]
        out_path = os.path.join(args.out_dir, f"{match_id}_split_view.png")
        cv2.imwrite(out_path, combined_img)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()