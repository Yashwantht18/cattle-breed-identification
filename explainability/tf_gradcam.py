
import tensorflow as tf
import numpy as np
import cv2


class TFGradCAM:
    """
    Grad-CAM for a Functional model with a nested backbone (MobileNetV2, EfficientNet, etc.)

    Architecture: Input → Rescaling → backbone(4D) → GAP → BN → Dropout → Dense → ... → softmax
    
    Strategy (avoids nested-model graph KeyError):
      1. Extract the Rescaling layer, backbone, and head layers separately.
      2. In compute_heatmap:
         a. Run input through Rescaling.
         b. Run through backbone to get 4D feature map.
         c. tape.watch(feature_map) — then run head layers on top of it.
         d. Gradient of class score w.r.t. feature map → Grad-CAM heatmap.
      
      This works because the tape tracks the forward path from the watched
      feature_map tensor through the head layers, without needing to trace
      across model boundaries in a Functional graph.
    """

    def __init__(self, model_path):
        full_model = tf.keras.models.load_model(model_path, compile=False)

        self.rescaling_layer = None
        self.backbone        = None
        self.head_layers     = []
        backbone_found       = False

        for layer in full_model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            elif isinstance(layer, tf.keras.layers.Rescaling):
                self.rescaling_layer = layer
                print(f"[Grad-CAM] Rescaling layer: '{layer.name}'  "
                      f"scale={layer.scale}, offset={layer.offset}")
            elif isinstance(layer, tf.keras.Model):
                self.backbone = layer
                backbone_found = True
                print(f"[Grad-CAM] Backbone: '{layer.name}'  "
                      f"output_shape={layer.output_shape}")
            elif backbone_found:
                self.head_layers.append(layer)

        if self.backbone is None:
            raise ValueError("No backbone sub-model found in the model.")

        print(f"[Grad-CAM] Head layers: {[l.name for l in self.head_layers]}")
        print(f"[Grad-CAM] Initialised successfully.")

    # ------------------------------------------------------------------
    def compute_heatmap(self, img_array, pred_index=None):
        """
        img_array : np.ndarray  shape (1, H, W, 3), pixels in [0, 1].
        Returns   : 2-D float32 numpy array in [0, 1].
        """
        img = tf.cast(img_array, tf.float32)

        # 1. Rescale to backbone's expected range ([-1, 1] for MobileNetV2)
        if self.rescaling_layer is not None:
            rescaled = self.rescaling_layer(img, training=False)
        else:
            rescaled = img

        # 2. Quick forward pass to determine predicted class if not given
        if pred_index is None:
            feat_q = self.backbone(rescaled, training=False)
            x_q    = feat_q
            for layer in self.head_layers:
                x_q = layer(x_q, training=False)
            pred_index = int(tf.argmax(x_q[0]))

        # 3. Grad-CAM forward pass
        #    tape.watch is called on feature_map (a computed tensor, not a variable).
        #    Since class_score is then computed FROM feature_map through head_layers,
        #    the gradient d(class_score)/d(feature_map) can be computed.
        with tf.GradientTape() as tape:
            feature_map = self.backbone(rescaled, training=False)  # (1, H, W, C)
            tape.watch(feature_map)

            x = feature_map
            for layer in self.head_layers:
                x = layer(x, training=False)

            class_score = x[:, pred_index]  # scalar

        grads = tape.gradient(class_score, feature_map)  # (1, H, W, C)

        if grads is None:
            print("[Grad-CAM] WARNING: gradient is None — returning blank heatmap.")
            h, w = feature_map.shape[1], feature_map.shape[2]
            return np.zeros((h, w), dtype=np.float32)

        # 4. Pool gradients across spatial dims → per-channel importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)

        # 5. Weight each feature map channel and average → spatial heatmap
        weighted = feature_map[0] * pooled_grads               # (H, W, C)
        heatmap  = tf.reduce_mean(weighted, axis=-1)            # (H, W)

        # 6. ReLU + normalise to [0, 1]
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)

        if max_val == 0:
            return np.zeros(heatmap.shape, dtype=np.float32)

        return (heatmap / max_val).numpy()


# -----------------------------------------------------------------------
def apply_heatmap(heatmap, original_img, alpha=0.45):
    """
    Overlay a Grad-CAM heatmap (float [0,1]) on an original BGR image (cv2).
    Returns a blended BGR uint8 image.
    """
    h, w = original_img.shape[:2]
    heatmap_r = cv2.resize(heatmap, (w, h))
    heatmap_u = np.uint8(255 * heatmap_r)
    colored   = cv2.applyColorMap(heatmap_u, cv2.COLORMAP_JET)  # blue=low, red=high
    blended   = colored * alpha + original_img * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)
