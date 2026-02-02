import os
import torch
import torch.nn as nn
from torchvision import models

def init_demo():
    print("Initializing demo environment...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Create dummy classes.txt with full Indian breed list
    classes = [
        'Gir', 'Sahiwal', 'Red Sindhi', 'Tharparkar', 'Kankrej', 'Deoni', 'Rathi', 
        'Amritmahal', 'Hallikar', 'Kangayam', 'Khillari', 'Nagori', 'Ponwar', 
        'Malnad Gidda', 'Alambadi', 'Bargur', 'Pulikulam', 'Umblachery', 'Hariana', 
        'Ongole', 'Siri', 'Kenkatha', 'Mewati', 'Kathani', 'Sanchori', 'Masilum', 
        'Karan Fries', 'Gaolao', 'Malvi', 'Nimari', 'Bachaur', 'Binjharpuri', 
        'Ghumsuri', 'Khariar', 'Motu', 'Purabi', 'Shahabadi', 'Belahi', 'Hissar', 
        'Jersind', 'Sunandini', 'Vechur', 'Krishna Valley', 'Punganur', 'Badri', 
        'Dagri', 'Kosali', 'Lakhimi', 'Nari', 'Red Kandhari', 'Gangatiri', 
        'Kenwariya', 'Lohani', 'Purnea'
    ]
    with open('models/classes.txt', 'w') as f:
        f.write('\n'.join(classes))
    
    # Create a dummy model (untrained ResNet-50)
    model_path = 'models/cattle_resnet50.pth'
    if not os.path.exists(model_path):
        print("Creating dummy model weights...")
        model = models.resnet50(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(classes))
        torch.save(model.state_dict(), model_path)
    
    print("Demo initialization complete.")

if __name__ == "__main__":
    init_demo()
