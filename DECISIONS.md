# Technical Decisions Log

This document records only major, project-defining technical decisions, their practical reasons, and trade-offs.

---

## Decision 1: Single-Channel Input for SAR Imagery
- **What was chosen**: Load SAR images as single-channel tensors `[1, 256, 256]` instead of 3-channel RGB `[3, 256, 256]`.
- **Why**: SAR sensors capture single-frequency microwave backscatter. Dataset inspection proved that R, G, and B channels are 100% identical duplicates. Slicing to 1 channel saves 66% memory and cuts initial layer convolutional parameters by 3x.
- **Trade-off**: Cannot directly plug into standard ImageNet-pretrained backbones without modifying the first layer, but perfectly suited for our custom network.
- **Syllabus Concept**: Input channels, convolutional kernels, parameter efficiency.

---

## Decision 2: Thresholding Anti-Aliased Masks at $\ge 128$
- **What was chosen**: Binarize all mask targets using `(mask >= 128).float()`.
- **Why**: The dataset audit discovered that 1,847 training masks contained smoothed edge values ($1 \dots 254$) from author resizing, while validation masks were strictly $\{0, 255\}$. Thresholding at 128 standardizes all ground truth into crisp binary targets.
- **Trade-off**: Discards soft boundary gradients in favor of clear ground-truth labels required for binary cross-entropy.
- **Syllabus Concept**: Ground-truth target representation for binary classification.

---

## Decision 3: Path-Scoped Split Isolation
- **What was chosen**: Match images and masks strictly within their parent directory (`data/images/train/` with `data/masks/train/`, and `val/` with `val/`).
- **Why**: All 1,615 filenames in `val` (`sentinel_0.png`, etc.) share identical names with files in `train`, but contain completely different images. Merging folders or pairing by filename alone would cause catastrophic data overwriting.
- **Trade-off**: Code must track split paths explicitly rather than using a single flat folder.
- **Syllabus Concept**: Dataset split integrity and preventing validation contamination.

---

## Decision 4: Lightweight Custom U-Net (482K Parameters)
- **What was chosen**: Build a 3-level Encoder-Decoder CNN with skip connections from scratch (`SimpleSegmentationCNN`), totaling 481,745 trainable parameters.
- **Why**: Heavy pretrained models (e.g. ResNet-50 with 25M+ parameters) act as black boxes and are prone to overfitting on cropped SAR patches. A custom 482K-parameter model trains fast on any hardware and is easy to explain layer-by-layer during a viva. Skip connections recover the sharp oil-slick boundaries lost during pooling.
- **Trade-off**: Slightly lower theoretical capacity than multi-million parameter models, but fully sufficient for binary segmentation on $256 \times 256$ patches.
- **Syllabus Concept**: Encoder-decoder architecture, spatial pooling, upsampling, and skip connections.

---

## Decision 5: `BCEWithLogitsLoss` as Initial Objective
- **What was chosen**: Use PyTorch's `nn.BCEWithLogitsLoss` combining raw logits with Binary Cross Entropy.
- **Why**: Combines Sigmoid activation and BCE loss into a single operation using the mathematical log-sum-exp identity. Prevents arithmetic underflow (`NaN` values) that occurs when calculating $\log(\sigma(z))$ separately on extreme positive/negative values.
- **Trade-off**: Predictions require an explicit `torch.sigmoid()` during evaluation, but training is mathematically stable.
- **Syllabus Concept**: Loss functions, binary cross-entropy, and numerical stability.

---

## Decision 6: Adam Optimizer with $\alpha = 0.001$
- **What was chosen**: Adam optimizer with initial learning rate $10^{-3}$, $\beta_1 = 0.9$, and $\beta_2 = 0.999$.
- **Why**: Combines Momentum (first-order moving averages) and RMSprop (adaptive per-parameter learning rates based on squared gradients). Converges much faster and more reliably than standard Stochastic Gradient Descent (SGD) without requiring manual learning rate scheduling.
- **Trade-off**: Requires $2\times$ parameter memory to store momentum and variance buffers, but easily fits our 482K-parameter network.
- **Syllabus Concept**: Optimizers, learning rate, gradient descent, momentum, and adaptive learning rates.

---

## Decision 7: Global Pixel Aggregation for Segmentation Metrics
- **What was chosen**: Sum True Positives (TP), False Positives (FP), and False Negatives (FN) across all 105.8 million validation pixels before calculating Precision, Recall, IoU, and Dice score.
- **Why**: 4.71% of our validation patches contain pure ocean with zero oil spill pixels ($TP + FN = 0$). Calculating per-patch Dice and averaging causes division by zero or introduces arbitrary smoothing constant bias. Global aggregation provides mathematically exact, unskewed metrics.
- **Trade-off**: Large oil slicks have a higher weight in the final score than tiny oil slicks, which is mathematically representative of total spill volume.
- **Syllabus Concept**: Evaluation metrics, confusion matrix, and class imbalance in segmentation.

---

## Decision 8: Transparent Labeling as "Validation Metrics" (No Test Set Fabricated)
- **What was chosen**: Explicitly label evaluation output as "Validation Metrics", not "Test Metrics".
- **Why**: The checkpoint was chosen using validation loss during training. The Zenodo archive provides only `train` and `val` directories. Claiming this is an independent test set would be academically dishonest.
- **Trade-off**: Acknowledges that performance is measured on the tuning split rather than a truly unseen third partition.
- **Syllabus Concept**: Train / Validation / Test split methodology and avoiding optimistic bias.

---

## Decision 9: 4-Panel Qualitative Visualization with Red Overlay
- **What was chosen**: Output a 4-panel comparison per sample: (1) Raw SAR image, (2) Ground Truth, (3) Predicted Mask with sample IoU/Dice, and (4) Red Prediction Overlay.
- **Why**: Numerical scores alone do not explain *how* the model fails or succeeds. Red overlays immediately reveal whether errors stem from boundary blur, speckle noise false alarms, or missed thin slicks.
- **Trade-off**: Saves static PNG files rather than an interactive web UI, but produces clean, portable figures ready for insertion into report documents and presentation slides.
- **Syllabus Concept**: Qualitative error analysis, visual model interpretability, and boundary localization.

---

## Decision 10: Training-Only Geometric Data Augmentation
- **What was chosen**: Random horizontal flips ($p=0.5$), vertical flips ($p=0.5$), and random $90^\circ$ rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) applied identically to both the input SAR image and its ground-truth mask. Augmentation is strictly training-only and never applied to validation data.
- **Why**: SAR satellite imagery captures the sea surface from an overhead nadir viewpoint. Oil slicks, surface waves, and ocean currents have no canonical upright orientation. Geometric augmentation quadruples scene variation without changing pixel intensity distributions, preventing the CNN from memorizing localized spatial coordinates (combating overfitting).
- **Trade-off**: Requires synchronizing geometric transforms across image-mask pairs, but runs with negligible CPU overhead via NumPy vector operations.
- **Syllabus Concept**: Data augmentation, rotational invariance, and preventing overfitting.

---

## Decision 11: Regularization via Weight Decay and Early Stopping
- **What was chosen**: Integrated Adam L2 weight decay (configurable, e.g. $10^{-4}$) and validation-loss early stopping with configurable patience (default $N=3$).
- **Why**: Deep convolutional networks can overfit to specific high-frequency speckle patterns. Weight decay penalizes excessively large parameter weights, ensuring smooth feature detectors. Early stopping halts training automatically if validation loss plateaus, guaranteeing the saved checkpoint is the optimal generalization state.
- **Trade-off**: May stop training before the maximum epoch count is reached, which saves compute and preserves generalization.
- **Syllabus Concept**: Overfitting, L2 regularization (weight decay), generalization, and early stopping.

