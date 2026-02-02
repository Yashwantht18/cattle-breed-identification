import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os

class BovineClassifier:
    def __init__(self, model_path, class_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if os.path.exists(class_path):
            with open(class_path, 'r') as f:
                self.classes = f.read().splitlines()
        else:
            self.classes = ['Gir', 'Sahiwal', 'Red Sindhi', 'Tharparkar', 'Kankrej', 'Murrah', 'Jaffrabadi', 'Surti']
            
        num_classes = len(self.classes)
        
        self.breed_to_type = {
            'Gir': 'Cattle', 'Sahiwal': 'Cattle', 'Red Sindhi': 'Cattle', 'Tharparkar': 'Cattle', 
            'Kankrej': 'Cattle', 'Ongole': 'Cattle', 'Hariana': 'Cattle', 'Hallikar': 'Cattle',
            'Murrah': 'Buffalo', 'Jaffrabadi': 'Buffalo', 'Surti': 'Buffalo', 'Mehsana': 'Buffalo',
            'Bhadawari': 'Buffalo', 'Nili-Ravi': 'Buffalo', 'Pandharpuri': 'Buffalo'
        }
        
        self.model = models.resnet50(weights='DEFAULT')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Error loading model weights: {e}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_animal_type(self, breed):
        return self.breed_to_type.get(breed, "Unknown")

    def predict(self, image_path, confidence_threshold=60.0):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        top1_conf = top3_prob[0].item() * 100
        top2_conf = top3_prob[1].item() * 100
        margin = top1_conf - top2_conf
        
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9)).item()
        
        is_bovine = True
        
        if top1_conf < confidence_threshold:
            is_bovine = False
        
        if entropy > 1.3: 
            is_bovine = False
            
        if top1_conf < 75 and margin < 15:
            is_bovine = False

        results = []
        for i in range(3):
            breed_idx = top3_idx[i].item()
            if breed_idx < len(self.classes):
                breed_name = self.classes[breed_idx]
                results.append({
                    "breed": breed_name,
                    "animal_type": self.get_animal_type(breed_name),
                    "confidence": float(top3_prob[i].item()) * 100
                })
            
        return results, input_tensor, is_bovine

if __name__ == "__main__":
    pass

