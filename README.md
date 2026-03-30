# Multi-Modal Emotion Classifier

A Streamlit-based web application for detecting and classifying multiple emotions from text input using machine learning.

## Features

- **Multi-Label Classification**: Detect multiple emotions simultaneously from text input
- **Real-Time Detection**: Get instant emotion predictions as you type
- **User-Friendly Interface**: Clean, intuitive UI built with Streamlit
- **Confidence Scores**: View probability scores for each emotion

## Supported Emotions

The classifier can detect multiple emotions including (but not limited to):
- Joy/Happiness
- Sadness
- Anger
- Fear
- Surprise
- And more emotional states

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-emotion-classifier.git
cd multimodal-emotion-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter text in the input field to classify emotions

## Project Structure

```
multimodal-emotion-classifier/
├── app.py                    # Main Streamlit application
├── complete_pipeline.joblib  # Trained model pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
└── .gitignore                # Git ignore rules
```

## Requirements

- Python 3.8+
- Streamlit >= 1.28.0
- joblib >= 1.3.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Machine learning powered by scikit-learn
