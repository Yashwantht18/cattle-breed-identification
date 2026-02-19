
import tensorflow as tf
import numpy as np
import cv2

class TFGradCAM:
    def __init__(self, model_path, last_conv_layer_name=None):
        full_model = tf.keras.models.load_model(model_path)
        
        # Check if the model is wrapped in Sequential and has a functional base model as first layer
        # This matches our training script structure: Sequential([base_model, pooling, dropout, dense])
        if isinstance(full_model.layers[0], tf.keras.Model):
            self.base_model = full_model.layers[0]
            self.classifier_layers = full_model.layers[1:]
        else:
            self.base_model = full_model
            self.classifier_layers = []

        if last_conv_layer_name is None:
            # For MobileNetV2, the last relevant conv layer is usually 'Out_relu' or similar
            # Iterating to find the last Conv2D or ReLU
            for layer in reversed(self.base_model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break
        
        print(f"Using layer {last_conv_layer_name} for Grad-CAM")
        self.last_conv_layer_name = last_conv_layer_name
        
        # Create a model that maps input -> [last_conv_output, prediction]
        # We need to reconstruct the forward pass
        last_conv_layer = self.base_model.get_layer(self.last_conv_layer_name)
        
        # Model 1: Input -> Last Conv
        self.conv_model = tf.keras.Model(self.base_model.inputs, last_conv_layer.output)
        
        # Model 2: Last Conv -> Prediction
        # This is tricky with functional API if we don't rebuild. 
        # Easier approach: simple Grad-CAM using gradient of output wrt conv layer
        # But we need the whole chain.
        
        # Let's use the `tf.GradientTape` on the full model, but watch the intermediate tensor.
        self.full_model = full_model

    def compute_heatmap(self, img_array, pred_index=None):
        with tf.GradientTape() as tape:
            # 1. Get the output of the last convolutional layer
            conv_outputs = self.conv_model(img_array)
            tape.watch(conv_outputs)
            
            # 2. Pass this output through the rest of the classifier layers
            x = conv_outputs
            # Apply global average pooling which might be implicitly done by MobileNetV2 base? 
            # No, in our training script: Sequential([base_model, pooling, dropout, dense])
            # But wait, base_model (MobileNetV2) output is 4D (batch, 7, 7, 1280).
            # So we need to apply the subsequent layers in order.
            
            for layer in self.classifier_layers:
                x = layer(x)
            
            predictions = x
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

def apply_heatmap(heatmap, original_img, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * alpha + original_img
    
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)
