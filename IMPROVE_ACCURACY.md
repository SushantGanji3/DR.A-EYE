# Guide: Improving Model Accuracy from 82% to 97%

## Current Status
- **Current Accuracy:** 82%
- **Target Accuracy:** 97%
- **Improvement Needed:** +15%

## Strategies Implemented

### 1. Class Imbalance Handling ⚖️

**Problem:** Dataset has severe class imbalance (No_DR: 1805, Severe: 193)

**Solutions:**
- **Weighted Loss Function:** Use class weights in CrossEntropyLoss
- **Weighted Random Sampler:** Oversample minority classes during training
- **Focal Loss:** Focuses learning on hard examples

```python
# Class weights automatically computed
class_weights = compute_class_weight('balanced', classes, labels)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

### 2. Advanced Data Augmentation 🎨

**Enhanced Augmentation:**
- Random Crop (256→224) for better generalization
- More aggressive color jitter (brightness, contrast, saturation, hue)
- Random Affine transformations
- Gaussian Blur
- Random Erasing
- **Mixup Augmentation:** Mixes two images for regularization

```python
# Mixup combines two images
mixed_x = λ * x1 + (1-λ) * x2
loss = λ * loss(pred, y1) + (1-λ) * loss(pred, y2)
```

### 3. Better Training Strategy 📈

**Optimizer Improvements:**
- **AdamW** instead of Adam (better weight decay)
- Lower initial learning rate (0.0001 vs 0.001)
- **OneCycleLR** scheduler for optimal learning rate schedule
- Gradient clipping to prevent exploding gradients

**Training Parameters:**
- More epochs (100 vs 50)
- Higher patience (10 vs 7)
- Better early stopping criteria

### 4. Model Architecture Options 🏗️

**Current:** ResNet-18

**Alternatives to try:**
1. **ResNet-50** - Deeper network, better feature extraction
2. **EfficientNet-B3** - Better accuracy/efficiency tradeoff
3. **Vision Transformer (ViT)** - State-of-the-art architecture
4. **Ensemble Models** - Combine multiple models

### 5. Additional Techniques 🚀

**Regularization:**
- Increased weight decay (0.01)
- Label smoothing (optional)
- Dropout layers (if needed)

**Advanced Methods:**
- **Test-Time Augmentation (TTA):** Average predictions from multiple augmented versions
- **Progressive Resizing:** Start with smaller images, gradually increase
- **Pseudo-labeling:** Use model predictions on unlabeled data
- **Knowledge Distillation:** Train smaller model from larger teacher

## How to Run Improved Training

### Option 1: Use the Improved Training Script

```bash
cd model
python train_improved.py
```

### Option 2: Use Jupyter Notebook

Open `notebooks/03_improved_training.ipynb` and run all cells.

### Option 3: Custom Training

```python
from model.train_improved import train_improved_model

train_improved_model(
    base_path='../data/raw/DiabeticRetinopathyDataset',
    num_epochs=100,
    batch_size=32,
    learning_rate=0.0001,
    use_focal_loss=True,
    use_mixup=True,
    use_weighted_sampler=True
)
```

## Expected Improvements

| Technique | Expected Gain |
|-----------|---------------|
| Class Weighted Loss | +3-5% |
| Advanced Augmentation | +2-4% |
| Mixup | +2-3% |
| Better Optimizer/Scheduler | +2-3% |
| Longer Training | +1-2% |
| **Total Expected** | **+10-17%** |

## Monitoring Progress

Watch for:
- **Validation accuracy** should steadily increase
- **Training/validation gap** should be small (no overfitting)
- **Learning rate** follows cosine schedule
- **Class-wise accuracy** improves for minority classes

## Troubleshooting

### If accuracy plateaus:
1. Try different learning rates (0.00005, 0.0002)
2. Increase mixup alpha (0.3, 0.4)
3. Try different architectures
4. Use ensemble methods

### If overfitting:
1. Increase data augmentation
2. Add more dropout
3. Increase weight decay
4. Use more regularization

### If underfitting:
1. Train longer (more epochs)
2. Increase model capacity
3. Reduce regularization
4. Use larger batch size

## Next Steps After Training

1. **Evaluate on test set** - Get final accuracy
2. **Compare with baseline** - Measure improvement
3. **Analyze confusion matrix** - See which classes improved
4. **Update API** - Use new model for predictions
5. **Document results** - Update README with new accuracy

## Advanced: Ensemble Methods

For even better accuracy, combine multiple models:

```python
# Train multiple models with different seeds
models = [train_model(seed=42), train_model(seed=123), train_model(seed=456)]

# Average predictions
predictions = (model1(x) + model2(x) + model3(x)) / 3
```

Expected gain: +2-5% additional accuracy

## Timeline

- **Training Time:** ~4-6 hours on CPU, ~30-60 min on GPU
- **Expected Result:** 90-97% accuracy
- **Best Case:** 97%+ with ensemble

## Files Created

- `model/best_resnet18_improved.pth` - Improved model
- `model/training_history_improved.json` - Training metrics
- `model/confusion_matrix_improved.png` - Confusion matrix
- `model/training_history_improved.png` - Training plots

