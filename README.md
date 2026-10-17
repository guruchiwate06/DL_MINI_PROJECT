# Oil Spill Segmentation in SAR Satellite Images

A Deep Learning mini-project for binary oil-spill segmentation on Synthetic Aperture Radar (SAR) satellite imagery using PyTorch.

---

## 1. Project Directory Structure
```
DL_MINI_PROJECT/
├── data/
│   ├── images/ (train: 6,455, val: 1,615)
│   └── masks/  (train: 6,455, val: 1,615)
├── src/
│   ├── dataset.py            # Dataset loader and normalization
│   ├── model.py              # SimpleSegmentationCNN (481K params)
│   ├── train.py              # Training loop with validation checkpointing
│   ├── evaluate.py           # Quantitative validation metrics
│   └── visualize.py          # 4-panel visual prediction comparisons
├── outputs/
│   ├── checkpoints/          # best_model.pth (saved weights)
│   └── visualizations/       # 4-panel comparison plots (.png)
├── README.md                 # Project guide and execution instructions
├── DECISIONS.md              # Key technical decisions and trade-offs
├── FLOW.md                   # Execution flow and tensor dimensions
└── VIVA_NOTES.md             # Syllabus concepts, viva Q&A, and results explanation
```

---

## 2. Dataset Summary
- **Source**: Refined Deep-SAR Oil Spill (SOS) dataset from Zenodo.
- **Total Images**: 8,070 pairs (6,455 train, 1,615 val), 100% paired, 0 missing.
- **Resolution**: Uniform $256 \times 256$ pixels.
- **Channels**: 1-channel grayscale SAR backscatter intensity in $[0.0, 1.0]$.
- **Mask Encoding**: Binary $\{0.0, 1.0\}$ ($0$ = Sea Background, $1$ = Oil Spill).
- **Class Balance**: 72.95% Sea ($77.2\text{M}$ px) vs. 27.05% Oil ($28.6\text{M}$ px).

---

## 3. Verified Validation Metrics (Trained Model Checkpoint)

Evaluated across all **1,615 validation images** ($105,840,640$ pixels) using `outputs/checkpoints/best_model.pth` (Epoch 10, GPU training with data augmentation & weight decay):

| Metric | Final Model (`best_model.pth`) | Baseline (`baseline_best_model.pth`) | Comparison / Note |
| :--- | :---: | :---: | :--- |
| **Validation Loss** | **0.2615** | 0.2807 | **-0.0192** (Cross-entropy error reduced by 6.8%) |
| **Training Loss** | **0.2356** | 0.3247 | Steady convergence, no overfitting divergence |
| **Precision** | **80.74%** | 77.89% | **+2.85%** (1.33M fewer false positive pixels) |
| **Recall (Sensitivity)** | **76.65%** | 80.80% | More conservative boundary delineation |
| **IoU (Jaccard Index)** | **64.81%** | 65.72% | Area of overlap divided by area of union |
| **Dice Coefficient (F1)** | **78.65%** | 79.31% | Global harmonic mean of precision and recall |
| **Evaluated Images** | **1,615 / 1,615** | 1,615 / 1,615 | 100% of validation split evaluated |
| **Checkpoint Path** | `outputs/checkpoints/best_model.pth` | `outputs/checkpoints/baseline_best_model.pth` | Both checkpoints preserved |

*Note: All results represent **Validation Metrics** because the checkpoint was chosen based on minimum validation loss. An independent third test split was not provided in the original Zenodo dataset.*

---

## 4. Environment & Execution Commands

### Dedicated CUDA Virtual Environment
A dedicated virtual environment [`.venv`](file:///.venv) is configured with CUDA-enabled PyTorch (`torch 2.6.0+cu124`):
```powershell
# Activate the dedicated project environment:
.venv\Scripts\activate

# Or run commands directly using:
.venv\Scripts\python.exe <script.py>
```

### Installation Method (for Reference)
```powershell
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install matplotlib
```

### Verify CUDA Acceleration
Run this quick command to verify GPU availability:
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0), '| VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))"
```
*Expected Output:*
```text
CUDA available: True | GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU | VRAM (GB): 6.0
```

---

### Execution Commands (Demonstration Guide)

Run these commands in order from the project root inside the activated environment:

```bash
# 1. Verify Dataset Loading & Geometric Augmentation:
python src/dataset.py

# 2. Verify Model Architecture & Forward Pass:
python src/model.py

# 3. Train Model with GPU Acceleration, Live Terminal Feedback, Early Stopping, and Augmentation:
# Options: --lr 0.001 (or 0.0003), --augment (training-only flips/rotations), --weight_decay 0.0001, --patience 3
python -u src/train.py --epochs 10 --batch_size 16 --lr 0.001 --augment --weight_decay 0.0001 --patience 3

# 4. Evaluate Checkpoint on Validation Split (Uses CUDA automatically):
python src/evaluate.py --checkpoint outputs/checkpoints/best_model.pth --threshold 0.5

# 5. Generate 4-Panel Visual Prediction Overlays:
python src/visualize.py --checkpoint outputs/checkpoints/best_model.pth --num_samples 5 --threshold 0.5
```

### How to Confirm Training Device is CUDA
When `src/train.py` starts, check the startup banner printed in the terminal:
```text
Device                   : cuda
```
If you wish to force CPU execution, pass the `--cpu` flag:
```bash
python -u src/train.py --cpu --smoke_test
```

### Live Terminal Progress During Training
When running `src/train.py`, the terminal displays:
- **Startup summary**: Device (`cuda`), parameter count, dataset sizes, batch size, learning rate, and regularizations.
- **Batch progress (every 10 batches)**: Batch index, completion percentage, running loss, and elapsed time.
- **Epoch summary**: Training loss, validation loss, validation Dice, validation IoU, and epoch duration.
- **Checkpoint updates**: Prints when a new minimum validation loss checkpoint is saved.
- **Early stopping**: Triggers and prints if validation performance fails to improve for the configured patience.

---

## 5. Limitations
1. **Georeferencing**: The cropped PNG patches do not contain acquisition dates or GPS coordinates, preventing ERA5 historical wind matching.
2. **Look-Alikes**: Natural low-wind areas form dark radar patches that may trigger occasional false alarms.
3. **VRAM Consideration**: The RTX 3050 GPU features 6 GB VRAM. Batch size 16 easily fits in under 1.5 GB VRAM; avoid unnecessarily huge batch sizes (>64) to prevent out-of-memory errors.