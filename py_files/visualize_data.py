import argparse
import os
import cv2
import numpy as np

def parse_match_file(path):
    """
    Returns mapping: match_id -> list of (bbox_xywh, cls).
    """
    by_id = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            
            mid = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            cls = int(parts[5])
            
            # Store as [x, y, w, h]
            by_id.setdefault(mid, []).append(([x, y, w, h], cls))
    return by_id

def draw_boxes(img, items, color, label_prefix):
    for mid, (box, cls) in items:
        x, y, w, h = map(int, box)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label_prefix}{mid} c{cls}", (x, max(10, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="Visualize Ground Truth Data")
    parser.add_argument("--base_dir", required=True, help="Path to cv_data_hw2")
    parser.add_argument("--matches_dir", required=True, help="Path to matched_annotations")
    parser.add_argument("--out_dir", default="vis_data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Scan files
    files = sorted([f for f in os.listdir(args.matches_dir) if f.endswith("_match.txt")])
    if not files:
        print("No match files found.")
        return

    print(f"Found {len(files)} files. Visualizing first 10...")
    
    for fname in files[:10]: # Limit to 10 for safety
        # Parse Filename: folder-img1-img2_match.txt
        stem = fname[:-10]
        parts = stem.split("-")
        if len(parts) < 3: continue
        
        folder = parts[0]
        img1_base = parts[1]
        img2_base = "-".join(parts[2:])
        
        # Load Images
        img1_path = os.path.join(args.base_dir, "data", folder, img1_base + ".jpg")
        img2_path = os.path.join(args.base_dir, "data", folder, img2_base + ".jpg")
        
        if not os.path.exists(img1_path): img1_path = img1_path.replace(".jpg", ".png")
        if not os.path.exists(img2_path): img2_path = img2_path.replace(".jpg", ".png")
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None: continue

        # Parse Boxes
        by_id = parse_match_file(os.path.join(args.matches_dir, fname))
        
        # Draw
        for mid, entries in by_id.items():
            if len(entries) < 2: continue # Need Old and New pos
            
            # Entry 0 = Initial (Blue)
            box_old, cls_old = entries[0]
            x, y, w, h = map(int, box_old)
            cv2.rectangle(img1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Entry 1 = Final (Green)
            box_new, cls_new = entries[-1]
            x, y, w, h = map(int, box_new)
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save Combined
        h = max(img1.shape[0], img2.shape[0])
        comb = np.zeros((h, img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
        comb[:img1.shape[0], :img1.shape[1]] = img1
        comb[:img2.shape[0], img1.shape[1]:] = img2
        
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}.png"), comb)
        print(f"Saved {stem}.png")

if __name__ == "__main__":
    main()