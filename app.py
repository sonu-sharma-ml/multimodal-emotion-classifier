"""
Multi-Emotion Classifier - Streamlit Frontend
A web application for detecting emotions in text using a multi-label classifier.
"""

import streamlit as st
import pandas as pd
import os
import base64
from utils import load_model, predict_emotions, process_batch

# Page configuration
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🌿",
    layout="wide"
)

# Load CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css()

# Emotion UI mappings
EMOTION_COLORS = {
    'anger': '#e53935',    # Red
    'fear': '#8e24aa',     # Purple
    'joy': '#ffd54f',      # Yellow/Gold
    'sadness': '#1e88e5',  # Blue
    'surprise': '#fb8c00'  # Orange
}

EMOTION_ICONS = {
    'anger': '😠',
    'fear': '😨',
    'joy': '😊',
    'sadness': '😢',
    'surprise': '😲'
}

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

def set_example_text(text):
    st.session_state.text_input = text

def generate_csv_download_link(df, filename="emotion_predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration:none;"><button class="stButton" style="background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%); color: white; border: none; padding: 0.6rem 2.5rem; border-radius: 8px; font-weight: 600; cursor: pointer; text-transform: uppercase; margin-top: 10px;">📥 Download Predictions CSV</button></a>'
    return href

def main():
    # Title
    st.markdown("<h1 class='title'>🌿 Neural Emotion Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Advanced Multi-Modal AI for Text Sentiment Analysis</p>", unsafe_allow_html=True)
    
    # Load model
    try:
        pipeline = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
        
    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Analyze Text", "📁 Batch Upload", "📜 Session History"])
    
    # ---------------------------------------------------------
    # TAB 1: Analyze Text
    # ---------------------------------------------------------
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Enter Text to Analyze")
            text_input = st.text_area(
                "Type or paste text:",
                value=st.session_state.text_input,
                height=160,
                placeholder="e.g., 'I am so thrilled with this new update!'"
            )
            
            analyze_button = st.button("🔍 Analyze Emotions", type="primary")
            
        with col2:
            st.markdown("### 💡 Quick Examples")
            examples = [
                "I'm so happy to see you! This is wonderful!",
                "I'm terrified of what might happen next.",
                "This is absolutely ridiculous! I'm so angry!",
                "I feel so empty and alone right now.",
                "Wow! I never expected that to happen!"
            ]
            
            for i, example in enumerate(examples):
                st.button(f"Example {i+1}", key=f"ex_{i}", on_click=set_example_text, args=(example,))
                
        # Handle Analysis
        if analyze_button:
            if not text_input.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            else:
                try:
                    detected_emotions, probabilities, labels = predict_emotions(text_input, pipeline)
                    
                    # Log to history
                    st.session_state.history.append({
                        'text': text_input,
                        'detected': detected_emotions,
                        'probabilities': dict(zip(labels, probabilities))
                    })
                    
                    st.markdown("---")
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("### 📊 Prediction Results")
                    
                    if detected_emotions:
                        emojis = " ".join([EMOTION_ICONS[e] for e in detected_emotions])
                        names = ", ".join([e.capitalize() for e in detected_emotions])
                        st.success(f"**Primary Detected:** {emojis} {names}")
                    else:
                        st.info("ℹ️ No strong emotions detected (below threshold).")
                    
                    st.markdown("<br/>", unsafe_allow_html=True)
                    
                    # Columns for emotion cards
                    cols = st.columns(len(labels))
                    
                    for idx, (col, emotion) in enumerate(zip(cols, labels)):
                        prob = probabilities[idx]
                        color = EMOTION_COLORS.get(emotion, '#4caf50')
                        icon = EMOTION_ICONS.get(emotion, '❓')
                        is_detected = emotion in detected_emotions
                        
                        # Style specific to detection status
                        border_style = f"border-color: {color}; box-shadow: 0 0 15px {color}40;" if is_detected else ""
                        opacity = "1.0" if is_detected else "0.3"
                        
                        with col:
                            st.markdown(f'''
                            <div class="emotion-card" style="{border_style} opacity: {opacity};">
                                <h3>{icon}</h3>
                                <strong>{emotion.capitalize()}</strong>
                                <p class="emotion-prob" style="color: {color};">{prob*100:.1f}%</p>
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    st.markdown("<br/>### 📈 Probability Distribution", unsafe_allow_html=True)
                    for emotion, prob in zip(labels, probabilities):
                        color = EMOTION_COLORS.get(emotion, '#4caf50')
                        st.markdown(f"<span style='color:{color}; font-weight:bold;'>{emotion.capitalize()}: {prob*100:.1f}%</span>", unsafe_allow_html=True)
                        st.progress(prob)
                        
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

    # ---------------------------------------------------------
    # TAB 2: Batch Upload
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### 📁 Batch CSV Processing")
        st.markdown("Upload a CSV file containing a column of text to analyze multiple sentences at once.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(df)}** rows.")
                
                # Select column
                text_col = st.selectbox("Select the column containing text:", df.columns)
                
                if st.button("🚀 Process Batch"):
                    with st.spinner("Analyzing emotions..."):
                        result_df = process_batch(df, text_col, pipeline)
                        st.success("✅ Processing complete!")
                        
                        with st.expander("Preview Results", expanded=True):
                            st.dataframe(result_df)
                            
                        # Download link
                        st.markdown(generate_csv_download_link(result_df, f"emotions_processed_{uploaded_file.name}"), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error reading/processing file: {e}")

    # ---------------------------------------------------------
    # TAB 3: History
    # ---------------------------------------------------------
    with tab3:
        st.markdown("### 📜 Session History")
        
        if not st.session_state.history:
            st.info("Your analysis history for this session will appear here.")
        else:
            if st.button("🗑️ Clear History", key="clear_history"):
                st.session_state.history = []
                st.rerun()
                
            for idx, item in enumerate(reversed(st.session_state.history)):
                detected_names = ", ".join([e.capitalize() for e in item['detected']])
                badges = " ".join([EMOTION_ICONS[e] for e in item['detected']])
                
                st.markdown(f'''
                <div class="history-item">
                    <div class="history-text">"{item['text']}"</div>
                    <div class="history-labels">
                        {"" if not item['detected'] else "Detected: " + badges + " " + detected_names}
                        {"" if item['detected'] else "Detected: None"}
                    </div>
                </div>
                ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()