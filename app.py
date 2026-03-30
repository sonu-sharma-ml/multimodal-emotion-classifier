"""
Multi-Emotion Classifier - Streamlit Frontend
A web application for detecting emotions in text using a multi-label classifier.
"""

import streamlit as st
import joblib
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

# Load the model and label columns
@st.cache_resource
def load_model():
    """Load the trained model and label columns."""
    model_path = os.path.join(os.path.dirname(__file__), 'complete_pipeline.joblib')
    pipeline = joblib.load(model_path)
    return pipeline

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        padding: 20px;
    }
    .emotion-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Emotion colors mapping
EMOTION_COLORS = {
    'anger': '#e74c3c',
    'fear': '#9b59b6',
    'joy': '#f1c40f',
    'sadness': '#3498db',
    'surprise': '#e67e22'
}

EMOTION_ICONS = {
    'anger': '😠',
    'fear': '😨',
    'joy': '😊',
    'sadness': '😢',
    'surprise': '😲'
}

def main():
    """Main function to run the Streamlit app."""
    
    # Title
    st.markdown("<h1 class='title'>🎭 Multi-Emotion Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#7f8c8d;'>Detect multiple emotions from text using AI</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    try:
        pipeline = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Input section
    st.subheader("📝 Enter Your Text")
    
    text_input = st.text_area(
        "Type or paste text to analyze:",
        height=150,
        placeholder="e.g., 'I can't believe I got the promotion! This is amazing!'"
    )
    
    # Analyze button
    if st.button("🔍 Analyze Emotions", type="primary"):
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            # Make prediction
            try:
                vectorizer = pipeline['vectorizer']
                model = pipeline['model']
                labels = pipeline['labels']
                
                # Transform text and predict
                text_features = vectorizer.transform([text_input])
                prediction = model.predict(text_features)[0]
                
                # Get probabilities - use decision_function with sigmoid as fallback
                try:
                    probabilities = model.predict_proba(text_features)[0]
                except AttributeError:
                    # Version mismatch - use decision function with sigmoid
                    import numpy as np
                    scores = model.decision_function(text_features)[0]
                    probabilities = 1 / (1 + np.exp(-scores))
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Create results container
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                
                # Show detected emotions
                detected_emotions = [labels[i] for i in range(len(labels)) if prediction[i] == 1]
                
                if detected_emotions:
                    st.success(f"**Detected Emotions:** {', '.join([EMOTION_ICONS[e] + ' ' + e.capitalize() for e in detected_emotions])}")
                else:
                    st.info("No strong emotions detected in the text.")
                
                st.markdown("### Emotion Probabilities")
                
                # Create columns for each emotion
                cols = st.columns(5)
                
                for idx, (col, emotion) in enumerate(zip(cols, labels)):
                    prob = probabilities[idx]
                    color = EMOTION_COLORS.get(emotion, '#95a5a6')
                    icon = EMOTION_ICONS.get(emotion, '❓')
                    
                    with col:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, {color}22, {color}44);
                            padding: 15px;
                            border-radius: 10px;
                            text-align: center;
                            border: 2px solid {color};
                        ">
                            <h3 style="margin:0;">{icon}</h3>
                            <strong>{emotion.capitalize()}</strong>
                            <p style="font-size:24px; margin:10px 0; color:{color};">
                                {prob*100:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Progress bar visualization
                st.markdown("### 📈 Probability Distribution")
                for emotion, prob in zip(labels, probabilities):
                    color = EMOTION_COLORS.get(emotion, '#95a5a6')
                    st.progress(prob)
                    st.markdown(f"<span style='color:{color}; font-weight:bold;'>{emotion.capitalize()}: {prob*100:.1f}%</span>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Model info
                st.markdown("---")
                st.markdown("""
                <div style='text-align:center; color:#7f8c8d; font-size:12px;'>
                    Model: OneVsRest Logistic Regression with TF-IDF Features<br>
                    Trained on 22,741 samples | Micro F1: 0.688
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
    
    # Sidebar with info
    st.sidebar.title("ℹ️ About")
    st.sidebar.info("""
    **Multi-Emotion Classifier**
    
    This app uses a trained machine learning model to detect 5 emotions in text:
    
    - 😠 Anger
    - 😨 Fear
    - 😊 Joy
    - 😢 Sadness
    - 😲 Surprise
    
    The model uses TF-IDF vectorization and OneVsRest Logistic Regression to perform multi-label classification.
    """)
    
    st.sidebar.title("📋 Example Texts")
    examples = [
        "I'm so happy to see you! This is wonderful!",
        "I'm terrified of what might happen next.",
        "This is absolutely ridiculous! I'm so angry!",
        "I feel so empty and alone right now.",
        "Wow! I never expected that to happen!"
    ]
    
    for i, example in enumerate(examples):
        if st.sidebar.button(f"Example {i+1}", key=f"ex_{i}"):
            st.session_state.text_input = example
    
    # Handle session state for example buttons
    if 'text_input' in st.session_state:
        st.text_area("Example loaded:", value=st.session_state.text_input, height=100, key="example_display")

if __name__ == "__main__":
    main()