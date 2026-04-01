"""
Utility functions for ML Model loading and Batch Processing
"""
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner="Loading ML Pipeline...")
def load_model():
    """Load the trained model pipeline from disk."""
    model_path = os.path.join(os.path.dirname(__file__), 'complete_pipeline.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    return joblib.load(model_path)

def predict_emotions(text, pipeline):
    """Predict emotions and probabilities for a single text input."""
    vectorizer = pipeline['vectorizer']
    model = pipeline['model']
    labels = pipeline['labels']
    
    # Vectorize
    text_features = vectorizer.transform([text])
    
    # Predict binary outcome
    prediction = model.predict(text_features)[0]
    
    # Predict probabilities (with fallback for models lacking predict_proba natively)
    try:
        probabilities = model.predict_proba(text_features)[0]
    except AttributeError:
        scores = model.decision_function(text_features)[0]
        # Sigmoid function
        probabilities = 1 / (1 + np.exp(-scores))
        
    # Get labels where prediction is 1
    detected = [labels[i] for i in range(len(labels)) if prediction[i] == 1]
    
    return detected, probabilities, labels

def process_batch(df, text_column, pipeline):
    """
    Process a pandas DataFrame column of texts and append 
    the model predictions and probabilities as new columns.
    """
    texts = df[text_column].astype(str).tolist()
    
    vectorizer = pipeline['vectorizer']
    model = pipeline['model']
    labels = pipeline['labels']
    
    # Transform all texts at once for efficiency
    features = vectorizer.transform(texts)
    
    # Predict exactly as in single text function
    predictions = model.predict(features)
    
    try:
        probabilities = model.predict_proba(features)
    except AttributeError:
        scores = model.decision_function(features)
        probabilities = 1 / (1 + np.exp(-scores))
        
    result_df = df.copy()
    
    # Collate predicted text labels
    pred_labels = []
    for row in predictions:
        row_labels = [labels[i] for i in range(len(labels)) if row[i] == 1]
        pred_labels.append(", ".join(row_labels))
        
    result_df['Detected_Emotions'] = pred_labels
    
    # Append individual probabilities to help users understand thresholds
    for i, label in enumerate(labels):
        result_df[f'Prob_{label.capitalize()}'] = np.round(probabilities[:, i], 4)
        
    return result_df
