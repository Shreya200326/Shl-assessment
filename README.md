# Grammar Scoring Engine from Voice Samples

## Overview

This project implements a simple and interpretable grammar scoring system for spoken audio responses. The goal is to take short voice recordings as input and predict a continuous grammar score, similar to how human evaluators rate grammatical quality.

The solution follows an end to end pipeline that converts speech to text, extracts linguistic features from the transcript, and trains a regression model to estimate grammar scores.

---

## Approach

1. **Speech to Text**
   Audio files are transcribed using a pretrained Wav2Vec2 model. This allows the system to work directly with raw speech without manual transcription.

2. **Feature Extraction**
   From the generated text, simple and interpretable linguistic features are extracted, such as word count, average word length, sentence count, and short word ratio. These features capture structural patterns related to grammatical quality.

3. **Model Training**
   A Random Forest regressor is trained on a subset of the training data to balance performance and computational efficiency. The model learns to map linguistic features to human annotated grammar scores.

4. **Prediction and Submission**
   The trained model is used to predict grammar scores for unseen test samples, and results are saved in a submission ready CSV file.

---

## Project Structure

```
├── grammar_scoring_engine.py
├── submission.csv
└── README.md
```

---

## Design Decisions

* A pretrained ASR model was used instead of training from scratch to ensure robustness and save time.
* Simple linguistic features were chosen to keep the model interpretable and easy to explain.
* A subset of the data was used during training to manage runtime constraints common in hiring assessments.

---

## How to Run

1. Ensure the dataset is available in the expected directory structure.
2. Install the required Python dependencies.
3. Run `grammar_scoring_engine.py` to train the model and generate `submission.csv`.

---

## Summary

This project demonstrates a practical and explainable approach to grammar scoring from speech by combining speech recognition, basic natural language processing, and classical machine learning techniques.
