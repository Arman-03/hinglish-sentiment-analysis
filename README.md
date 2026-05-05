# Hinglish Sentiment Analysis

A comprehensive sentiment analysis project for Hinglish (Hindi-English code-mixed) e-commerce reviews. This project fine-tunes a pre-trained BERT model on synthetic Hinglish data to classify sentiments into Positive, Negative, and Neutral categories.

## Authors
- **K N Akthar** (2023101007)
- **Vanshika Ahlawat** (2023113003)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data](#data)
- [Training Pipeline](#training-pipeline)
- [Model Details](#model-details)
- [Demo App](#demo-app)
- [Results](#results)
- [Usage](#usage)


## Overview
This project implements a sentiment classifier for Hinglish text, commonly found in Indian e-commerce platforms like Flipkart and Meesho. The pipeline includes data exploration, baseline comparison, fine-tuning a Hinglish-specific BERT model, and deployment via a Streamlit web app.

The model achieves high accuracy on synthetic test data and provides a practical demo for real-world Hinglish sentiment analysis.

## Features
- **Data Exploration**: Comprehensive EDA with class distribution, text length analysis, and word clouds
- **Baseline Models**: Zero-shot classification using BART and Groq Llama-3
- **Fine-tuned Model**: HingBERT (l3cube-pune/hing-bert) fine-tuned for 3-class sentiment classification
- **Evaluation**: Detailed metrics, confusion matrices, and error analysis
- **Web Demo**: Interactive Streamlit app for sentiment prediction
- **Hugging Face Integration**: Model hosted on Hugging Face Hub for easy access

## Installation

### Prerequisites
- Python 3.8+
- Git
- (Optional) Google Colab for training (as shown in the notebook)

### Clone the Repository
```bash
git clone https://github.com/your-username/hinglish-sentiment-analysis.git
cd hinglish-sentiment-analysis
```

### Install Dependencies
For the training notebook:
```bash
pip install transformers datasets torch scikit-learn accelerate wordcloud matplotlib seaborn groq nltk
```

For the demo app:
```bash
cd hugging_face
pip install -r requirements.txt
```

## Data
The dataset consists of synthetic Hinglish e-commerce reviews generated using Google's Gemini AI. Each review is labeled as POSITIVE, NEGATIVE, or NEUTRAL.

- **Location**: `data/datav2.json`
- **Format**: JSON array with `text` and `label` fields
- **Size**: ~1000+ samples (stratified across classes)
- **Example**:
```json
{
  "text": "Samsung Galaxy S24 ka review dena chahta hoon - superb hai bhai. Roz use kar raha hoon, abhi tak koi issue nahi. Must buy!",
  "label": "POSITIVE"
}
```

## Training Pipeline
The complete training pipeline is documented in `training_notebook.ipynb`:

1. **Setup**: Install packages, set seeds, configure device
2. **Data Loading**: Load and preprocess the JSON dataset
3. **EDA**: Analyze class distribution, text lengths, top words, word clouds
4. **Data Split**: 80/10/10 stratified train/val/test split
5. **Baseline**: BART zero-shot classification
6. **Main Model**: Fine-tune HingBERT with early stopping
7. **Upper Bound**: Groq Llama-3 zero-shot on subset
8. **Evaluation**: Compare models, confusion matrices, error analysis
9. **Export**: Save model and upload to Hugging Face

### Key Hyperparameters
- Model: `l3cube-pune/hing-bert`
- Max Length: 128 tokens
- Batch Size: 32 (train), 64 (eval)
- Learning Rate: 2e-5
- Epochs: 5 (with early stopping)
- Optimizer: AdamW with weight decay

## Model Details
- **Base Model**: HingBERT - BERT pre-trained on Hinglish corpus by L3Cube Pune
- **Task**: Sequence Classification (3 classes)
- **Architecture**: BERT + Linear classification head
- **Parameters**: ~110M total, ~110M trainable
- **Hugging Face Repo**: [Arman-03/sentiment-analysis-hinglish](https://huggingface.co/Arman-03/sentiment-analysis-hinglish)

## Demo App
An interactive web application built with Streamlit for real-time sentiment analysis.

### Features
- Text input for Hinglish reviews
- Real-time prediction with confidence scores
- Example reviews for testing
- Visual feedback with emojis and progress bars
- Interpretation notes

### Running the App
```bash
cd hugging_face
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Results
Based on evaluation on held-out test set:

| Model | Accuracy | Macro F1 | Training Time |
|-------|----------|----------|---------------|
| BART Zero-Shot | 0.75 | 0.74 | None |
| HingBERT Fine-tuned | 0.92 | 0.91 | ~5 min (T4 GPU) |
| Groq Llama-3 (n=100) | 0.88 | 0.87 | None (API) |

### Key Insights
- Fine-tuned HingBERT significantly outperforms zero-shot baselines
- Model performs best on Positive reviews, struggles slightly with Neutral
- High-confidence predictions are usually correct (>90% accuracy)
- Common errors: Sarcasm detection, mixed sentiments

## Usage

### Python Inference
```python
from transformers import pipeline

# Load the model
classifier = pipeline(
    "text-classification",
    model="Arman-03/sentiment-analysis-hinglish"
)

# Predict sentiment
result = classifier("Bahut achha product hai!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.95}]
```

### Command Line
```bash
# For the app
cd hugging_face
streamlit run app.py
```

---

**Note**: This project was developed as part of SMAI Assignment 3. The synthetic data was generated using Google's Gemini AI for demonstration purposes.