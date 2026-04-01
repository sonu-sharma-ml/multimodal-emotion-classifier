import os
import joblib
import torch
import torch.nn as nn
import timm

original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    kwargs['map_location'] = 'cpu'
    kwargs['weights_only'] = False
    return original_torch_load(f, **kwargs)

torch.load = patched_torch_load

class FERModel(nn.Module):
    def __init__(self, num_classes=7, backbone_name='mobilenetv3_large_100'):
        super(FERModel, self).__init__()
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=False, 
            num_classes=0,
            global_pool=''
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 48, 48)
            features = self.backbone(dummy_input)
            num_features = features.shape[1]
        self.backbone.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

try:
    path = 'fer_complete_package_20260401_141226.pkl'
    print("Loading...")
    package = joblib.load(path)
    print("Loaded package keys:", package.keys())
    
    config = package.get('config', {})
    labels = package.get('class_labels', [])
    print("Config:", config)
    print("Labels:", labels)
    
    print(package.get('model', 'No model key'))
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
