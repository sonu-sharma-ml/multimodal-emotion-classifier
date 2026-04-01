"""
Facial Expression Recognition (FER) - Streamlit App
A web application for detecting emotions from facial images using deep learning.
"""

import streamlit as st
import cv2
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from PIL import Image
import timm

# IMPORTANT: Patch torch BEFORE importing anything else that might use it
# This is required to load models trained on CUDA on CPU-only machines
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    kwargs['map_location'] = 'cpu'
    kwargs['weights_only'] = False
    return original_torch_load(f, **kwargs)

torch.load = patched_torch_load


# Try to import mediapipe for face detection
try:
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
    from mediapipe import Image as mp_image, ImageFormat
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.warning("MediaPipe not available. Using OpenCV Haar Cascade for face detection.")


# Page configuration
st.set_page_config(
    page_title="FER - Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)


# Load CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css()


# Emotion mappings for FER (7 basic emotions)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_COLORS = {
    'angry': '#e53935',     # Red
    'disgust': '#7cb342',   # Light Green
    'fear': '#8e24aa',      # Purple
    'happy': '#ffd54f',     # Yellow/Gold
    'sad': '#1e88e5',       # Blue
    'surprise': '#fb8c00',  # Orange
    'neutral': '#90a4ae'    # Gray
}

EMOTION_ICONS = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
    'neutral': '😐'
}


# Define FERModel class based on actual model architecture
class FERModel(nn.Module):
    """PyTorch Neural Network for Facial Emotion Recognition using MobileNetV3 backbone"""
    def __init__(self, num_classes=7, backbone_name='mobilenetv3_small_100'):
        super(FERModel, self).__init__()
        
        # Use timm to create backbone
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=False, 
            num_classes=0,  # Remove classifier
            global_pool=''
        )
        
        # Get backbone output features
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 48, 48)
            features = self.backbone(dummy_input)
            num_features = features.shape[1]
        
        # Custom classifier for 7 emotions
        self.backbone.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


class FERPipeline:
    """Wrapper class for FER model pipeline"""
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.labels = EMOTION_LABELS
        self.config = None
        
    def load(self):
        """Load the trained model and configuration"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load the complete package
        package = joblib.load(self.model_path)
        
        if isinstance(package, dict):
            self.config = package.get('config', {})
            self.labels = package.get('class_labels', EMOTION_LABELS)
            
            # Try using the pre-instantiated model from the package if it exists
            if 'model' in package and isinstance(package['model'], torch.nn.Module):
                self.model = package['model']
                self.model.to('cpu')
            else:
                # Create model instance fallback
                self.model = FERModel(num_classes=len(self.labels))
                
                # Load state dict
                if 'model_state_dict' in package:
                    state_dict = package['model_state_dict']
                    # Handle DataParallel prefix if present
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            k = k[7:]  # Remove 'module.' prefix
                        new_state_dict[k] = v
                    self.model.load_state_dict(new_state_dict, strict=False)
            
            self.model.eval()
            
        return self
        
    def preprocess_image(self, img_array, target_size=(48, 48)):
        """Preprocess image for the model - expects RGB 3-channel input"""
        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:  # Single channel with batch dim
            img_array = cv2.cvtColor(img_array.squeeze(), cv2.COLOR_GRAY2RGB)
        
        # Resize to model input size
        resized = cv2.resize(img_array, target_size)
        
        # Normalize (ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (resized / 255.0 - mean) / std
        
        # Convert to tensor format (batch, channels, height, width)
        tensor = torch.FloatTensor(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor
        
    def predict(self, img_array):
        """Predict emotion from face image"""
        if self.model is None:
            self.load()
            
        # Preprocess
        tensor = self.preprocess_image(img_array)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(outputs, dim=1)[0].item()
            
        return prediction, probabilities.numpy()


@st.cache_resource(show_spinner="Loading FER Model...")
def load_fer_model():
    """Load the FER model"""
    # Try different model paths
    model_paths = [
        'fer_complete_package_20260401_141226.pkl',
        'fer_model.pkl',
        'models/fer_complete_package_20260401_141226.pkl',
        'models/fer_model.pkl'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            pipeline = FERPipeline(path)
            pipeline.load()
            return pipeline
    
    raise FileNotFoundError("No FER model found!")


def detect_faces_opencv(image):
    """Detect faces using OpenCV Haar Cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces


def detect_faces_mediapipe(image):
    """Detect faces using MediaPipe"""
    if not MEDIAPIPE_AVAILABLE:
        return []
    
    # Create MediaPipe image
    rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_img = mp_image(image_format=ImageFormat.SRGB, data=rgb)
    
    # Initialize detector
    model_path = 'models/blaze_face_short_range.tflite'
    if not os.path.exists(model_path):
        model_path = 'blaze_face_short_range.tflite'
    if not os.path.exists(model_path):
        return []
    
    base_options = base_options_module.BaseOptions(model_asset_path=model_path)
    options = FaceDetectorOptions(base_options=base_options)
    
    with FaceDetector.create_from_options(options) as detector:
        result = detector.detect(mp_img)
        
    faces = []
    if result.detections:
        h, w = image.shape[:2]
        for detection in result.detections:
            bbox = detection.bounding_box
            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height
            faces.append((x, y, width, height))
    
    return faces


def main():
    # Title
    st.markdown("<h1 class='title'>😊 Facial Emotion Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Face Expression Analysis</p>", unsafe_allow_html=True)
    
    # Load model
    try:
        model = load_fer_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please ensure the FER model file is in the correct location.")
        return
    
    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "📷 Webcam Capture", "ℹ️ About"])
    
    # ---------------------------------------------------------
    # TAB 1: Image Upload
    # ---------------------------------------------------------
    with tab1:
        st.markdown("### Upload an Image for Emotion Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg', 'bmp', 'webp']
            )
            
            st.markdown("""
            **Tips for best results:**
            - Use clear, front-facing photos
            - Single face images work best
            - Good lighting improves accuracy
            - Supported formats: PNG, JPG, JPEG, BMP, WEBP
            """)
        
        with col2:
            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                image_array = np.array(image.convert('RGB'))
                
                # Display original image
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Process button
                if st.button("🔍 Analyze Emotions", type="primary"):
                    with st.spinner("Detecting faces and analyzing emotions..."):
                        process_image(image_array, model)
    
    # ---------------------------------------------------------
    # TAB 2: Webcam Capture
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### Capture from Webcam")
        
        # Webcam input
        img_file = st.camera_input("Take a photo")
        
        if img_file is not None:
            # Convert to numpy array
            image = Image.open(img_file)
            image_array = np.array(image.convert('RGB'))
            
            # Display
            st.image(image, caption="Captured Image", use_container_width=True)
            
            # Process button
            if st.button("🔍 Analyze Captured Image", type="primary"):
                with st.spinner("Detecting faces and analyzing emotions..."):
                    process_image(image_array, model)
    
    # ---------------------------------------------------------
    # TAB 3: About
    # ---------------------------------------------------------
    with tab3:
        st.markdown("""
        ### About This Application
        
        This Facial Emotion Recognition (FER) system uses deep learning to detect 
        emotions from facial expressions in images.
        
        **Supported Emotions:**
        """)
        
        # Display emotion cards
        cols = st.columns(len(EMOTION_LABELS))
        for idx, (col, emotion) in enumerate(zip(cols, EMOTION_LABELS)):
            with col:
                color = EMOTION_COLORS.get(emotion, '#4caf50')
                icon = EMOTION_ICONS.get(emotion, '❓')
                st.markdown(f'''
                <div class="emotion-card" style="border-color: {color};">
                    <h3>{icon}</h3>
                    <strong>{emotion.capitalize()}</strong>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown("""
        ---
        
        **Technical Details:**
        - Model: MobileNetV3 backbone (timm) with custom classifier
        - Training: FER2013 dataset
        - Face Detection: MediaPipe / OpenCV Haar Cascade
        - Framework: PyTorch, Streamlit
        
        **File Structure:**
        - `fer_complete_package_20260401_141226.pkl` - Complete model pipeline
        """)


def process_image(image_array, model):
    """Process image and detect emotions"""
    
    # Detect faces
    if MEDIAPIPE_AVAILABLE:
        try:
            faces = detect_faces_mediapipe(image_array)
        except:
            faces = detect_faces_opencv(image_array)
    else:
        faces = detect_faces_opencv(image_array)
    
    if len(faces) == 0:
        st.warning("⚠️ No faces detected in the image. Please try with a different image.")
        return
    
    st.success(f"✅ Detected {len(faces)} face(s)")
    
    # Process each face
    results = []
    image_with_boxes = image_array.copy()
    
    for idx, (x, y, w, h) in enumerate(faces):
        # Extract face ROI
        face_roi = image_array[y:y+h, x:x+w]
        
        # Predict emotion
        pred_idx, probs = model.predict(face_roi)
        
        # Get emotion label
        emotion = model.labels[pred_idx]
        confidence = probs[pred_idx]
        
        results.append({
            'face_idx': idx,
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': dict(zip(model.labels, probs)),
            'bbox': (x, y, w, h)
        })
        
        # Draw bounding box
        color = tuple(int(EMOTION_COLORS[emotion].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), color, 2)
        
        # Add label
        label = f"{emotion.capitalize()}: {confidence*100:.1f}%"
        cv2.putText(image_with_boxes, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display result image
    st.markdown("### 🎯 Detection Results")
    st.image(image_with_boxes, caption="Emotion Detection Results", use_container_width=True)
    
    # Display detailed results
    st.markdown("---")
    for result in results:
        emotion = result['emotion']
        confidence = result['confidence']
        probs = result['probabilities']
        
        color = EMOTION_COLORS.get(emotion, '#4caf50')
        icon = EMOTION_ICONS.get(emotion, '❓')
        
        st.markdown(f'<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"### {icon} Face {result['face_idx'] + 1}: **{emotion.capitalize()}** ({confidence*100:.1f}%)")
        
        # Show probability distribution
        st.markdown("#### Probability Distribution")
        prob_cols = st.columns(len(EMOTION_LABELS))
        
        for idx, (col, emo) in enumerate(zip(prob_cols, EMOTION_LABELS)):
            with col:
                prob = probs[emo]
                emo_color = EMOTION_COLORS.get(emo, '#4caf50')
                emo_icon = EMOTION_ICONS.get(emo, '❓')
                st.markdown(f'''
                <div class="emotion-card" style="opacity: 1.0; padding: 10px;">
                    <strong>{emo_icon} {emo.capitalize()}</strong>
                    <p class="emotion-prob" style="color: {emo_color};">{prob*100:.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")


if __name__ == "__main__":
    main()
