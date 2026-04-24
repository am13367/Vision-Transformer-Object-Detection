import os
import argparse  # <--- NEW IMPORT
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_distances

# --- ARGUMENT PARSING (THE FIX) ---
def get_args():
    parser = argparse.ArgumentParser()
    # Now we pass these paths from the Sbatch file!
    parser.add_argument('--data_dir', type=str, required=True, help="Path to cv_data_hw2")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save annotations")
    return parser.parse_args()

# --- LOAD MODEL GLOBALLY ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- HELPER FUNCTIONS ---
def parse_annotation_file(ann_path):
    objects = []
    if not os.path.exists(ann_path):
        # Fail silently or warn, but don't crash immediately if one file is missing
        return objects
    
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Robust parsing for 6 or 8 column formats
            try:
                if len(parts) >= 8:
                    x, y, w, h = map(float, parts[3:7])
                    obj_type = int(parts[7])
                elif len(parts) >= 6:
                    x, y, w, h = map(float, parts[1:5])
                    obj_type = int(parts[5])
                else:
                    continue 

                objects.append({
                    'bbox': [x, y, w, h],
                    'type': obj_type,
                    'centroid': [x + w/2, y + h/2]
                })
            except ValueError:
                continue
    return objects

def extract_features(img, objects):
    features = []
    h_img, w_img = img.shape[:2]
    for obj in objects:
        x, y, w, h = map(int, obj['bbox'])
        # Clip to image boundaries to prevent crashing
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_img - x), min(h, h_img - y)
        
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            features.append(np.zeros(512))
            continue
        
        inp = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(inp).cpu().squeeze().numpy()
        features.append(feat)
    return np.array(features) if features else np.array([])

def compute_cost(img1, img2, objs1, objs2):
    if not objs1 or not objs2: return np.zeros((0,0))
    feats1 = extract_features(img1, objs1)
    feats2 = extract_features(img2, objs2)
    if len(feats1) == 0 or len(feats2) == 0: return np.zeros((len(objs1), len(objs2)))
    
    vis_cost = cosine_distances(feats1, feats2)
    
    n1, n2 = len(objs1), len(objs2)
    cent_cost = np.zeros((n1, n2))
    for i, o1 in enumerate(objs1):
        for j, o2 in enumerate(objs2):
            d = np.linalg.norm(np.array(o1['centroid']) - np.array(o2['centroid']))
            cent_cost[i,j] = d
    if cent_cost.max() > 0: cent_cost /= cent_cost.max()
    return vis_cost + 0.5 * cent_cost

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1+w1, x2+w2) - xi
    hi = min(y1+h1, y2 + h2) - yi
    if wi <= 0 or hi <= 0: return 0
    inter = wi * hi
    union = (w1*h1) + (w2*h2) - inter
    return inter / union

def process_pair(data_dir, output_dir, p1, a1, p2, a2):
    # Construct FULL paths using data_dir
    img1_path = os.path.join(data_dir, p1)
    img2_path = os.path.join(data_dir, p2)
    ann1_path = os.path.join(data_dir, a1)
    ann2_path = os.path.join(data_dir, a2)

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        # print(f"Missing image: {p1} or {p2}") 
        return

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    objs1 = parse_annotation_file(ann1_path)
    objs2 = parse_annotation_file(ann2_path)
    
    if not objs1 or not objs2 or img1 is None or img2 is None: return

    cost = compute_cost(img1, img2, objs1, objs2)
    row_ind, col_ind = linear_sum_assignment(cost)
    
    matches = []
    for r, c in zip(row_ind, col_ind):
        iou = compute_iou(objs1[r]['bbox'], objs2[c]['bbox'])
        # COMPLIANT LOGIC: IoU=0 OR class mismatch
        if iou == 0 or (iou > 0 and objs1[r]['type'] != objs2[c]['type']):
            matches.append((objs1[r], objs2[c]))
    
    if matches:
        # Create output filename
        # Ensure we don't carry over directory slashes into the filename
        safe_p1 = os.path.basename(p1).replace('.png', '').replace('.jpg', '')
        safe_p2 = os.path.basename(p2).replace('.png', '').replace('.jpg', '')
        folder = os.path.basename(os.path.dirname(p1))
        
        filename = f"{folder}-{safe_p1}-{safe_p2}_match.txt"
        
        with open(os.path.join(output_dir, filename), 'w') as f:
            for mid, (o1, o2) in enumerate(matches):
                x,y,w,h = map(int, o1['bbox'])
                f.write(f"{mid} {x} {y} {w} {h} {o1['type']}\n")
                x,y,w,h = map(int, o2['bbox'])
                f.write(f"{mid} {x} {y} {w} {h} {o2['type']}\n")

def main():
    args = get_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    index_path = os.path.join(args.data_dir, 'index.txt')
    if not os.path.exists(index_path):
        print(f"CRITICAL ERROR: Index file not found at {index_path}")
        return
        
    print(f"Reading index from: {index_path}")
    with open(index_path, 'r') as f:
        lines = f.readlines()
        
    print(f"Processing {len(lines)} pairs...")
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) == 4:
            # We pass args.data_dir to helper functions now
            process_pair(args.data_dir, args.output_dir, *[p.strip() for p in parts])

if __name__ == "__main__":
    main()