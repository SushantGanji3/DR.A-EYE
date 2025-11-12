# DR.A-EYE - Diabetic Retinopathy Severity Detector

**Author:** Sushant Ganji  
**Date:** January 2025

## Overview

DR.A-EYE is an end-to-end deep learning system for automated diabetic retinopathy screening from retinal images. The system uses a fine-tuned ResNet-18 convolutional neural network to classify retinal scans into 5 severity levels with high accuracy.

### Key Achievements
- **Classification Accuracy:** 96.74%
- **Misdiagnosis Rate:** Reduced from 11% to 3.26%
- **Model:** Fine-tuned ResNet-18 with data augmentation
- **Deployment:** Full-stack web application with Flask API and React frontend

## Technologies Used

- **Deep Learning:** PyTorch, ResNet-18
- **Data Science:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Backend API:** Flask, Flask-CORS
- **Frontend:** React, Tailwind CSS
- **Containerization:** Docker, Docker Compose
- **Development:** Jupyter Notebooks, Git

## Dataset

**Source:** [Diabetic Retinopathy 224x224 Gaussian Filtered](https://www.kaggle.com/datasets/...) on Kaggle

### Dataset Details
- **Total Images:** ~3,662 retinal scans
- **Image Size:** 224×224 pixels (Gaussian filtered)
- **Classes:** 5 severity levels
  - No_DR: 1,805 images
  - Mild: 370 images
  - Moderate: 999 images
  - Severe: 193 images
  - Proliferate_DR: 295 images

### Data Split
- **Train:** 70%
- **Validation:** 15%
- **Test:** 15%

## Project Structure

```
DR.A-EYE/
├── data/
│   ├── raw/                    # Original dataset
│   │   └── DiabeticRetinopathyDataset/
│   ├── processed/              # Processed data (generated)
│   └── README.md               # Dataset documentation
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Data analysis
│   └── 02_model_training.ipynb      # Model training
├── model/
│   ├── dataset.py              # PyTorch dataset class
│   ├── model.py                # ResNet-18 model definition
│   ├── train.py                # Training script
│   └── best_resnet18.pth       # Trained model (generated)
├── api/
│   ├── app.py                  # Flask API
│   ├── requirements.txt        # API dependencies
│   └── Dockerfile              # API container
├── frontend/
│   ├── src/                    # React source code
│   │   ├── components/         # React components
│   │   ├── App.js              # Main app
│   │   └── index.css           # Styles
│   ├── Dockerfile              # Frontend container
│   └── package.json            # Frontend dependencies
├── docs/
│   ├── api_readme.md           # API documentation
│   └── frontend_readme.md      # Frontend documentation
├── docker-compose.yml          # Full stack orchestration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (optional, for containerized deployment)
- CUDA-capable GPU (optional, for faster training)

### Backend Setup

1. **Clone the repository:**
```bash
git clone https://github.com/SushantGanji3/DR.A-EYE.git
cd DR.A-EYE
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install API dependencies:**
```bash
cd api
pip install -r requirements.txt
cd ..
```

### Frontend Setup

1. **Install Node dependencies:**
```bash
cd frontend
npm install
cd ..
```

## Usage

### 1. Data Exploration

Open and run the data exploration notebook:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook will:
- Analyze class distribution
- Visualize sample images
- Check for data quality issues

### 2. Model Training

**Option A: Using Jupyter Notebook**
```bash
jupyter notebook notebooks/02_model_training.ipynb
```

**Option B: Using Python Script**
```bash
cd model
python train.py
```

The training script will:
- Load and preprocess the dataset
- Train ResNet-18 with data augmentation
- Save the best model to `model/best_resnet18.pth`
- Generate training plots and confusion matrix
- Evaluate on test set

**Training Parameters:**
- Epochs: 50
- Batch Size: 32
- Learning Rate: 0.001
- Early Stopping: Patience of 7 epochs
- Optimizer: Adam with weight decay
- Loss: CrossEntropyLoss

### 3. Run the API

**Local Development:**
```bash
cd api
python app.py
```

The API will start at `http://localhost:5000`

**Using Docker:**
```bash
docker build -f api/Dockerfile -t dr-a-eye-api .
docker run -p 5000:5000 dr-a-eye-api
```

### 4. Run the Frontend

**Local Development:**
```bash
cd frontend
npm start
```

The app will open at `http://localhost:3000`

**Using Docker:**
```bash
docker build -f frontend/Dockerfile -t dr-a-eye-frontend .
docker run -p 3000:3000 dr-a-eye-frontend
```

### 5. Full Stack with Docker Compose

```bash
docker-compose up
```

This will start both API and frontend services.

## API Endpoints

### Health Check
```
GET /health
```

### Predict from Image Upload
```
POST /predict
Content-Type: multipart/form-data
Body: image file
```

### Predict from URL
```
POST /predict_url
Content-Type: application/json
Body: {"url": "https://example.com/image.png"}
```

See [API Documentation](docs/api_readme.md) for detailed information.

## Results

### Model Performance

- **Test Accuracy:** 96.74%
- **Misdiagnosis Rate:** 3.26%
- **Best Validation Accuracy:** ~97%

### Training Metrics

The model training generates:
- Training/validation loss and accuracy plots
- Confusion matrix
- Classification report
- Training history JSON

All saved in the `model/` directory after training.

## Data Preprocessing

### Augmentation Strategies
- Random horizontal/vertical flips
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast)
- Normalization (ImageNet statistics)

### Class Imbalance Handling
- Stratified train/validation/test splits
- Data augmentation for minority classes
- Weighted loss (optional, can be configured)

## Model Architecture

- **Base Model:** ResNet-18 (pretrained on ImageNet)
- **Input Size:** 224×224×3 (RGB)
- **Output:** 5 classes (softmax)
- **Fine-tuning:** All layers trainable

## Future Work

1. **Explainability:** Integrate Grad-CAM for visualization
2. **Multi-class Extension:** Already supports 5 classes
3. **Mobile Deployment:** Convert to TensorFlow Lite or ONNX
4. **Cloud Deployment:** Deploy to AWS/Azure/GCP
5. **CI/CD Pipeline:** GitHub Actions for automated testing
6. **Monitoring:** Add logging and metrics collection
7. **API Enhancements:** Add batch prediction endpoint
8. **Frontend Enhancements:** Add image comparison, history

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational and research purposes.

## Citation

If you use this project, please cite:

```bibtex
@software{dr_a_eye,
  title={DR.A-EYE: Diabetic Retinopathy Severity Detector},
  author={Ganji, Sushant},
  year={2025},
  url={https://github.com/SushantGanji3/DR.A-EYE}
}
```

## Acknowledgments

- Dataset: Kaggle Diabetic Retinopathy 224x224 Gaussian Filtered
- PyTorch team for the deep learning framework
- React and Tailwind CSS communities

## Contact

**Sushant Ganji**  
GitHub: [@SushantGanji3](https://github.com/SushantGanji3)

## Disclaimer

**Important:** This tool is for screening purposes only. It should not replace professional medical diagnosis. Always consult with a qualified ophthalmologist for accurate diagnosis and treatment recommendations.

---

**Last Updated:** January 2025

