"""
Script to load and inspect the FER complete package - find classifier
"""
import sys

# Patch torch.load at the module level first
import torch

_original_torch_load = torch.load

def _cpu_load(f, map_location=None, **kwargs):
    kwargs['map_location'] = 'cpu'
    kwargs['weights_only'] = False
    return _original_torch_load(f, **kwargs)

torch.load = _cpu_load

class FERModel(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(FERModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(256 * 6 * 6, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    import joblib
    
    model_path = 'fer_complete_package_20260401_141226.pkl'
    
    package = joblib.load(model_path)
    
    # Find classifier keys
    state_dict = package.get('model_state_dict', {})
    classifier_keys = [k for k in state_dict.keys() if 'head' in k or 'fc' in k or 'classifier' in k]
    print("Classifier keys:", classifier_keys)
    
    # Check last few keys
    print("\nLast 10 keys:")
    for k in list(state_dict.keys())[-10:]:
        print(f"  {k}: {state_dict[k].shape}")
    
    # Check config for model info
    config = package.get('config', {})
    print("\nConfig:", config)
    
    # Check class labels
    class_labels = package.get('class_labels', [])
    print("\nClass labels:", class_labels)