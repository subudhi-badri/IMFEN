import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    model = models.resnet18(weights=None)  
    model.fc = nn.Linear(model.fc.in_features, 2)  
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  
    model.eval()  
    return model

def predict_document(image_path, model):
    image = Image.open(image_path).convert('RGB')  
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():  
        outputs = model(image)  
        probs = torch.softmax(outputs, dim=1)  
        confidence, predicted_class = torch.max(probs, 1)  
        class_names = ['fake', 'real']  
        prediction = class_names[predicted_class.item()]
        return prediction, confidence.item()  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image')  
    parser.add_argument('--model-path', default='resnet18_model.pth', help='Path to trained model')  
    args = parser.parse_args()

    model = load_model(args.model_path)  
    prediction, confidence = predict_document(args.image, model)  
    print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")  
