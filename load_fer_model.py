"""
Script to load and inspect the FER complete package
"""
import pickle
import os
import cv2
import numpy as np

# Patch torch.cuda.is_available BEFORE importing joblib
import torch
original_cuda_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False

# Also patch torch.load to always map to cpu
original_torch_load = torch.load
def patched_torch_load(f, map_location=None, **kwargs):
    kwargs['map_location'] = 'cpu'
    kwargs['weights_only'] = False
    return original_torch_load(f, **kwargs)
torch.load = patched_torch_load

# Now define FERModel class - needs to be after torch patching
class FERModel:
    def __init__(self):
        self.model = None
        self.labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
    def load(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        import joblib
        package = joblib.load(model_path)
        if isinstance(package, dict):
            self.model = package.get('model', None)
            self.labels = package.get('labels', self.labels)
        return self
        
    def predict(self, img_array):
        if self.model is None:
            raise ValueError("Model not loaded")
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        features = normalized.flatten().reshape(1, -1)
        prediction = self.model.predict(features)[0]
        try:
            probabilities = self.model.predict_proba(features)[0]
        except:
            probabilities = np.zeros(len(self.labels))
            probabilities[prediction] = 1.0
        return prediction, probabilities


if __name__ == "__main__":
    import sys
    
    model_path = 'fer_complete_package_20260401_141226.pkl'
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        import joblib
        model = joblib.load(model_path)
        
        print("=== Model Package Contents ===")
        print(f"Type: {type(model)}")
        
        if isinstance(model, dict):
            print(f"Keys: {model.keys()}")
            for key, value in model.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif hasattr(value, '__len__'):
                    print(f"  {key}: len={len(value)}, type={type(value)}")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print(f"Model type: {type(model)}")
            if hasattr(model, 'classes_'):
                print(f"Classes: {model.classes_}")
            if hasattr(model, 'n_features_in_'):
                print(f"n_features_in: {model.n_features_in_}")
                
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()