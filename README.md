# 🌿 Neural Emotion Classifier

A premium, Streamlit-based web application for detecting and classifying multiple emotions from text input using machine learning.

## ✨ Features

- **Multi-Label Classification**: Accurately detect multiple emotions simultaneously from text input.
- **Batch CSV Processing**: Upload a CSV and analyze hundreds of rows at once, then download the full prediction results.
- **Session History**: Easily review your past analyses and their predicted emotions within a single session.
- **Premium Dark UI**: A visually stunning, modern glassmorphic interface that responds dynamically to user interactions.
- **Real-Time Confidence**: View precise probability scores distribution for each detected emotion category.
- **Face Detection (Beta)**: Includes a MediaPipe-powered real-time face detection module (`face_detection_new_api.py`) for visual modality exploration.

## 🎭 Supported Emotions

The Multi-Modal Classifier currently detects 5 primary emotional states:
- **Joy** 😊
- **Sadness** 😢
- **Anger** 😠
- **Fear** 😨
- **Surprise** 😲

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sonu-sharma-ml/multimodal-emotion-classifier.git
   cd multimodal-emotion-classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the App:** Open your browser and navigate to `http://localhost:8501`

## 📁 Project Architecture

```
multimodal-emotion-classifier/
├── app.py                    # Main Streamlit application and UI routing
├── utils.py                  # Core logic, model caching, and batch processing functions
├── style.css                 # Premium custom theme and glassmorphism styling
├── complete_pipeline.joblib  # Trained Scikit-Learn model pipeline
├── requirements.txt          # Python dependencies (Streamlit, Pandas, Scikit-Learn, Mediapipe)
├── test_batch.csv            # Sample CSV file for testing Bulk Process functionality
├── face_detection_new_api.py # Real-time Face Detection using MediaPipe Tasks API
├── app1.ipynb                # Experimental Jupyter Notebook for data exploration
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

## 🧠 Machine Learning Details

- **Feature Extraction**: TF-IDF Vectorization
- **Algorithm**: One-vs-Rest (OvR) Logistic Regression
- **Performance**: Capable of outputting raw probabilities (via `predict_proba` or mapped `decision_function` calculations depending on sklearn versions).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Bulk data processing via [pandas](https://pandas.pydata.org/)
