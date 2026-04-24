# Assignment 3: Moved Object Detection with DETR

**Name:** Ahmed Arkam Mohamed Faisaar  
**NetID:** am13367

---

## Overview
This repository contains an end-to-end pipeline for detecting moved objects in surveillance footage using the VIRAT dataset. The method implements **Option 2 (Pixel-wise Image Difference)**, feeding the absolute difference between two frames directly into a fine-tuned DETR (ResNet-50) model.

---

## File Structure
* `dataset.py`: Custom PyTorch dataset class. Computes `cv2.absdiff` between frame pairs and formats annotations for DETR.
* `model.py`: Utility functions to load the pre-trained `facebook/detr-resnet-50` model and image processors.
* `train.py`: Main training script using HuggingFace Trainer. Supports freezing specific layers for ablation studies.
* `evaluation.py`: Computes global Precision, Recall, and F1 scores using IoU matching (Threshold=0.5).
* `visualize_preds.py`: Generates side-by-side qualitative comparisons (Ground Truth vs. Predictions).
* `visualize_data.py`: Helper script used to verify the correctness of generated ground truth boxes.
* `data_ground_truth_labeller.py`: The provided script for generating annotations, **modified** to fix pathing issues on the cluster.
* `*.sbatch`: SLURM scripts for running the entire pipeline on the NYU HPC cluster.

---

## How to Run

### Step 1: Environment Setup
Before running any scripts, you must set up the Python environment and dependencies (PyTorch, Transformers, OpenCV, etc.) inside the Singularity overlay.

**1. Run the setup script:**
```bash
sbatch create_env.sbatch
```
**Note:** This script automatically handles the virtual environment creation, manual pip installation, and dependency downloads.

**2. Once the job finishes, activate the environment:**
```bash
source /scratch/am13367/venvs/assn3/bin/activate
```
Required for running helper scripts like `visualize_data.py` interactively.

---

### Step 2: Dataset Setup

**Important:** The `cv_data_hw2` dataset folder is EXCLUDED from this zip file to comply with submission size limits. An empty placeholder directory structure is provided instead.

#### Prerequisites
Before proceeding, ensure your main scratch directory (`/scratch/${USER}`) contains:
- `Ahmed_Arkam_Mohamed_Faisaar_CV_assignment3.zip` (this submission)
- `cv_data_hw2.zip` (the complete VIRAT dataset - provided separately)

#### Setup Steps
Run these commands from your main scratch directory (`/scratch/${USER}`):

```bash
# 1. Unzip the assignment code (This creates the assignment folder)
unzip Ahmed_Arkam_Mohamed_Faisaar_CV_assignment3.zip

# 2. Delete the empty data folder placeholder inside the submission
rm -rf Ahmed_Arkam_Mohamed_Faisaar_CV_assignment3/cv_data_hw2

# 3. Unzip the main data file (This creates the full cv_data_hw2 folder)
unzip cv_data_hw2.zip

# 4. Move the newly unzipped 'cv_data_hw2' data folder into the assignment directory
mv cv_data_hw2 Ahmed_Arkam_Mohamed_Faisaar_CV_assignment3/

# 5. Proceed into the assignment folder to start the pipeline
cd Ahmed_Arkam_Mohamed_Faisaar_CV_assignment3
```

#### Verification
After setup, your directory structure should look like:
```
Ahmed_Arkam_Mohamed_Faisaar_CV_assignment3/
├── py_files/
├── vis_predictions/
├── logs/
├── cv_data_hw2/          <-- Complete dataset
│   ├── data/             <-- Now contains all video subfolders
│   │   ├── VIRAT_S_000000/
│   │   ├── VIRAT_S_000001/
│   │   └── ...
│   └── index.txt         
├── data_ground_truth_labeller.py
└── ...
```

Verify the setup by checking:
```bash
ls cv_data_hw2/data/ | head -5  # Should show subfolder names like VIRAT_S_000000
```

#### Alternative: Using Symbolic Links
If `cv_data_hw2` already exists in your scratch directory:
```bash
# 1. Unzip the assignment
unzip Ahmed_Arkam_Mohamed_Faisaar_CV_assignment3.zip

# 2. Remove the placeholder
rm -rf Ahmed_Arkam_Mohamed_Faisaar_CV_assignment3/cv_data_hw2

# 3. Create a symbolic link (faster than copying)
ln -s /absolute/path/to/existing/cv_data_hw2 Ahmed_Arkam_Mohamed_Faisaar_CV_assignment3/cv_data_hw2

# 4. Navigate into the assignment
cd Ahmed_Arkam_Mohamed_Faisaar_CV_assignment3
```

---

### Step 3: Generate Ground Truth
Run the labeller to create the matched annotations required for training.
```bash
sbatch run_labeller.sbatch
```
**Output:** Creates a `matched_annotations/` folder containing processed `.txt` labels.

---

### Step 4: Training (Ablation Studies)
Four training modes are supported via the `--freeze_mode` argument in `train.py`.

**Important Note:** The training SBATCH scripts mount the Singularity overlay in read-only mode (`:ro`). This is intentional design to allow all four experiments to be submitted and trained simultaneously in parallel without locking the overlay file.

To run the experiments:
```bash
# 1. Train All Parameters (Best Performer)
sbatch run_train_all.sbatch

# 2. Train Transformer Only (High Recall)
sbatch run_train_transformer.sbatch

# 3. Train Backbone Only (Baseline/Failure Case)
sbatch run_train_backbone.sbatch

# 4. Train Class Head Only (Fast Convergence)
sbatch run_train_head.sbatch
```
**Output:** Checkpoints are saved to `checkpoints/<mode>/`.

---

### Step 5: Quantitative Evaluation
This script evaluates all valid checkpoints found in the directory and computes Precision/Recall/F1.
```bash
sbatch run_eval_all.sbatch
```
**Output:** Results are printed to the log file (e.g., `logs/eval_all_<jobid>.out`).

---

### Step 6: Qualitative Visualization
Generates PDFs showing Ground Truth (Green) vs. Model Predictions (Red) for random samples.
```bash
sbatch run_vis_combined.sbatch
```
**Output:** Visualizations are saved in `vis_predictions/<mode>/`.

---

## Summary of Results
* **Best Model:** The "All Params" model achieved the best balance (F1: 0.26), effectively learning to distinguish motion from static background.
* **Transformer Only:** Achieved the highest Recall (0.83) but lower Precision.
* **Backbone Only:** Failed to generalize effectively (Recall 0.43), indicating the pre-trained Transformer cannot interpret features from a modified backbone without co-adaptation.
* **Class Head Only:** Trained for only 10 epochs (vs 100) to prevent overfitting, as validation loss plateaued early (Epoch 7).

---

## Method Architecture

### Chosen Approach: Option 2 (Pixel-wise Image Difference)

**Rationale:**
This approach directly encodes motion information by computing the absolute pixel-wise difference between consecutive frames. The difference image highlights regions where objects have moved, providing a clear signal for the DETR model to focus on motion rather than static scene elements.

**Implementation Details:**
1. Both frames are resized to consistent dimensions (as required by DETR)
2. Pixel-wise absolute difference is computed using `cv2.absdiff(frame1, frame2)`
3. The difference image is fed directly into the pre-trained DETR model
4. The model learns to detect bounding boxes around regions with significant motion

---

## Modifications to Provided Code

**data_ground_truth_labeller.py:**
- Modified absolute file paths to work correctly on the NYU HPC cluster environment
- All core functionality (Hungarian matching, IoU filtering) remains unchanged
- Changes were necessary to ensure proper file I/O when running via SLURM

---

## Training Configuration

- **Model:** `facebook/detr-resnet-50` (pre-trained)
- **Dataset Split:** 80-20 train-test split
- **Epochs:** 100 (except Class Head Only: 10 epochs)
- **Batch Size:** 4
- **Learning Rate:** 1e-5
- **Optimizer:** AdamW with weight decay
- **Loss Function:** DETR's combined classification + bounding box regression loss

---

## Ablation Studies

Four fine-tuning strategies were evaluated:

1. **All Parameters:** Fine-tune the entire DETR model (backbone + transformer + head)
2. **Transformer Only:** Freeze ResNet-50 backbone, fine-tune transformer layers only
3. **Backbone Only:** Fine-tune ResNet-50 backbone, freeze transformer and head
4. **Class Head Only:** Fine-tune only the final classification head, freeze all other layers

**Key Findings:**
- Training all parameters achieved the best F1 score, balancing precision and recall
- Transformer-only training achieved highest recall but suffered from lower precision
- Backbone-only training failed to generalize, suggesting the transformer requires co-adaptation
- Class head training converged quickly but required early stopping to prevent overfitting

---

## Evaluation Metrics

**Precision:** Proportion of predicted boxes that correctly match ground truth objects (IoU ≥ 0.5)

**Recall:** Proportion of ground truth objects successfully detected by the model

**F1 Score:** Harmonic mean of precision and recall, providing a balanced performance measure

---

## Assumptions and Design Decisions

1. **IoU Threshold:** Used 0.5 as the standard threshold for matching predictions to ground truth
2. **Motion Definition:** Only objects with IoU = 0 between frames are considered "moved" (as per the provided labeller)
3. **Image Preprocessing:** All images are resized to DETR's expected input size before computing differences
4. **Class Head Epochs:** Limited to 10 epochs based on validation loss plateau at epoch 7
5. **Parallel Training:** Read-only overlay mounting enables simultaneous training of all ablation studies

---

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers (HuggingFace)
- OpenCV (cv2)
- NumPy
- Pillow
- Matplotlib (for visualization)
- Access to NYU HPC cluster with Singularity

All dependencies are automatically installed via `create_env.sbatch`.

---

## Contact

For questions or issues, please contact:
- **Name:** Ahmed Arkam Mohamed Faisaar
- **NetID:** am13367