import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        output[0, class_idx].backward()
        
        gradients = self.gradients
        activations = self.activations
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= np.max(heatmap)
        
        return heatmap

def apply_heatmap(heatmap, original_img):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + original_img
    superimposed_img = np.uint8(255 * (superimposed_img / np.max(superimposed_img)))
    
    return superimposed_img
