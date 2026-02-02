import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset_processing.preprocessing import prepare_data
import os

def train_model(data_dir, num_epochs=20, learning_rate=0.001, model_save_path='models/cattle_resnet50.pth', fine_tune=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader, classes = prepare_data(data_dir)
    num_classes = len(classes)
    
    # Save class names for inference
    with open('models/classes.txt', 'w') as f:
        f.write('\n'.join(classes))

    # Load pre-trained ResNet-50
    model = models.resnet50(weights='DEFAULT')
    
    if not fine_tune:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Freeze only the earlier layers, fine-tune the later ones (layer4 and fc)
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
    
    # Modify final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
        
        train_acc = train_corrects.double() / len(train_loader.dataset)
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        scheduler.step()
        
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved! (Val Loss: {val_loss:.4f})")
        
        print("-" * 10)

if __name__ == "__main__":
    # Train the model on the prepared mini-dataset
    train_model('data', num_epochs=5)
    print("Training complete.")
