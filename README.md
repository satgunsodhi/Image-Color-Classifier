# Pokemon Color Classifier

This project uses machine learning techniques to classify Pokemon by their dominant color. It employs both traditional machine learning (Random Forest) and deep learning (Convolutional Neural Network) approaches to predict the dominant color category of Pokemon images.

## Overview

The system analyzes Pokemon images, extracts dominant colors using K-means clustering, and classifies them into color categories. Two different models are implemented and compared:
1. A Random Forest classifier
2. A Convolutional Neural Network (CNN)

## Dependencies

The project requires the following Python libraries:
- jupyter
- ipywidgets
- huggingface-hub
- datasets
- pandas
- tensorflow
- keras
- opencv-python
- matplotlib
- scikit-learn
- seaborn

## Data Source

The project uses the "tungdop2/pokemon" dataset from Hugging Face, which contains Pokemon images and their corresponding names.

## Workflow

### 1. Preprocessing
- Load Pokemon images from the dataset
- Define color categories with corresponding hex values
- Create a function to find the closest color category for any RGB value using Euclidean distance

### 2. Feature Extraction
- Apply K-means clustering (k=3) to identify dominant colors in each Pokemon image
- Filter pixels using an alpha channel threshold to focus on the Pokemon rather than the background
- Map the dominant colors to predefined color categories

### 3. Traditional ML Model
- Flatten images to create feature vectors
- Encode color labels
- Split data into training and testing sets
- Train a Random Forest classifier
- Evaluate with accuracy metrics and confusion matrix
- Save the classification results to "Traditional_output.csv"

### 4. Convolutional Neural Network
- Build a CNN model with the following architecture:
  - Convolutional layer (32 filters, 3x3 kernel)
  - Max pooling layer (2x2)
  - Flatten layer
  - Dense layer (128 neurons)
  - Output layer (softmax activation)
- Split data into training, validation, and testing sets
- Train the model for 20 epochs
- Evaluate with accuracy, precision, and recall metrics
- Save the classification results to "CNN_output.csv"

## Output Files

The project generates three CSV files:
- `Dataset.csv` - The original dataset with Pokemon names and their true dominant colors
- `Traditional_output.csv` - Results from the Random Forest classifier
- `CNN_output.csv` - Results from the CNN model

## Color Categories

The system classifies Pokemon into 10 color categories:
- Red
- Pink
- Orange
- Yellow
- Purple
- Green
- Blue
- Brown
- White
- Gray

## Usage

The script is designed to be run in a Jupyter notebook environment. Each section of the code can be executed sequentially to reproduce the entire pipeline from data loading to model evaluation.

## Future Improvements

Potential enhancements to the project could include:
- Fine-tuning the CNN architecture for better performance
- Implementing data augmentation to improve model robustness
- Exploring more sophisticated color extraction techniques
- Adding a web interface for real-time Pokemon color classification