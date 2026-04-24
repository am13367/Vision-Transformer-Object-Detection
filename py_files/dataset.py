import os
from PIL import Image, ImageChops
import torch
from torch.utils.data import Dataset

class MatchedPairDataset(Dataset):
    """
    OPTION 2 Implementation (Matches Seba's Logic).
    - Input: Takes two images.
    - Process: Computes Pixel Difference (Img2 - Img1).
    - Output: Returns the Difference Image and the Bounding Box for Frame 2.
    """
    def __init__(self, base_dir, matches_dir):
        self.base_dir = base_dir
        self.matches_dir = matches_dir
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        if not os.path.exists(self.matches_dir):
            print(f"Error: {self.matches_dir} does not exist!")
            return samples

        # Scan for match files
        for fname in sorted(os.listdir(self.matches_dir)):
            if not fname.endswith("_match.txt"): continue

            # Robust Filename Parsing (Seba's method)
            stem = fname[:-10] # remove "_match.txt"
            parts = stem.split("-")
            
            if len(parts) < 3: continue
            
            # Reconstruct paths
            folder = parts[0]
            img1_base = parts[1]
            img2_base = "-".join(parts[2:])

            p1 = os.path.join(self.base_dir, "data", folder, img1_base + ".jpg")
            p2 = os.path.join(self.base_dir, "data", folder, img2_base + ".jpg")
            
            # Fallback to PNG if JPG doesn't exist
            if not os.path.exists(p1): p1 = p1.replace(".jpg", ".png")
            if not os.path.exists(p2): p2 = p2.replace(".jpg", ".png")

            if os.path.exists(p1) and os.path.exists(p2):
                samples.append({
                    "img1": p1, 
                    "img2": p2, 
                    "ann": os.path.join(self.matches_dir, fname)
                })
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load Images
        try:
            img1 = Image.open(sample["img1"]).convert("RGB")
            img2 = Image.open(sample["img2"]).convert("RGB")
        except Exception as e:
            # Safety skip if image is corrupt
            print(f"Error loading image: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # 2. Compute PIXEL DIFFERENCE (This is the Core of Option 2)
        # We subtract the images. Static background becomes black. Moved objects light up.
        diff_img = ImageChops.difference(img2, img1)

        # 3. Parse Annotations
        # We want the 'New' position (Frame 2) because that's what we are detecting.
        boxes = []
        labels = []
        
        # Dictionary trick: by_id[match_id] will keep overwriting until the last entry
        # Since our file writes (Old, New) in order, the last one in the dict will be New.
        by_id = {}
        
        if os.path.exists(sample["ann"]):
            with open(sample["ann"], "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6: continue
                    
                    match_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5]) # Top-Left format
                    cls = int(parts[5])
                    
                    # Convert xywh -> xyxy (x_min, y_min, x_max, y_max)
                    x_min = x
                    y_min = y
                    x_max = x + w
                    y_max = y + h
                    
                    by_id[match_id] = ([x_min, y_min, x_max, y_max], cls)

        for b, c in by_id.values():
            boxes.append(b)
            labels.append(c)

        # 4. Handle Empty Targets (Required for DETR stability)
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        target = {
            "boxes": boxes, 
            "labels": labels, 
            "image_id": torch.tensor([idx])
        }

        # Return the DIFFERENCE image and the target
        return diff_img, target

    def __len__(self):
        return len(self.samples)