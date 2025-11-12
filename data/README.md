# Dataset Documentation

## Source
Diabetic Retinopathy 224x224 Gaussian Filtered Dataset from Kaggle

## Dataset Structure
The dataset is located in `data/raw/DiabeticRetinopathyDataset/` and contains:

- **Images**: 224x224 pixel Gaussian filtered retinal images
- **Location**: `gaussian_filtered_images/gaussian_filtered_images/`
- **Classes**: 5 severity levels
  - `No_DR/`: No Diabetic Retinopathy (1,805 images)
  - `Mild/`: Mild DR (370 images)
  - `Moderate/`: Moderate DR (999 images)
  - `Severe/`: Severe DR (193 images)
  - `Proliferate_DR/`: Proliferative DR (295 images)

- **Labels**: `train.csv` contains id_code and diagnosis columns
  - Diagnosis mapping: 0=No_DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferate_DR

## Total Images
Approximately 3,662 images across 5 classes

## Train/Validation/Test Split
- Train: 70%
- Validation: 15%
- Test: 15%

## Preprocessing
Images are already resized to 224x224 and Gaussian filtered. Additional augmentation applied during training.

