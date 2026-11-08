# Dry Bean Classification — ML Assignment 2

## Problem Statement
Predict the variety of a dry bean using its geometrical measurements.

## Dataset
- Source: UCI Machine Learning Repository — Dry Bean
- Number of instances: 13611
- Number of features: 16
- Number of classes: 7

## Models
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

## Evaluation Metrics
Precision, Recall, and F1 Score use weighted averaging.
Multiclass AUC uses weighted one-vs-rest ROC AUC.

## Model Comparison

| Model                |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:---------------------|-----------:|-------:|------------:|---------:|-----------:|-------:|
| SVM                  |     0.9221 | 0.9937 |      0.9224 |   0.9221 |     0.9222 | 0.9059 |
| Logistic Regression  |     0.9207 | 0.9934 |      0.9215 |   0.9207 |     0.9209 | 0.9042 |
| KNN                  |     0.9166 | 0.9812 |      0.9174 |   0.9166 |     0.9168 | 0.8992 |
| Gaussian Naive Bayes |     0.8979 | 0.9902 |      0.9007 |   0.8979 |     0.8981 | 0.8773 |
| Decision Tree        |     0.8917 | 0.933  |      0.8912 |   0.8917 |     0.8913 | 0.8691 |

## Best Model
The model with the highest weighted F1 Score in this run is **SVM**.

## Observations
Top Performing Model: Support Vector Machine (SVM) achieved the highest overall performance with a weighted F1 Score of 0.9222 and an Accuracy of 0.9221.  
Close Second: Logistic Regression performed competitively right behind SVM, recording an Accuracy of 0.9207 and an F1 Score of 0.9209, indicating that a linear separation model fits this geometrical feature space very well.  
Full Ranking (by F1 Score and Accuracy):
SVM (F1: 0.9222 | Accuracy: 0.9221 | AUC: 0.9937)  
Logistic Regression (F1: 0.9209 | Accuracy: 0.9207 | AUC: 0.9934)  
K-Nearest Neighbors (KNN) (F1: 0.9168 | Accuracy: 0.9166 | AUC: 0.9812)  
Gaussian Naive Bayes (F1: 0.8981 | Accuracy: 0.8979 | AUC: 0.9902)  
Decision Tree (F1: 0.8913 | Accuracy: 0.8917 | AUC: 0.9330)

## Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Start the app: `streamlit run app.py`

## GitHub Repository
Replace this line with your GitHub repository URL.

## Live Streamlit Application
Replace this line with your Streamlit Community Cloud URL.
