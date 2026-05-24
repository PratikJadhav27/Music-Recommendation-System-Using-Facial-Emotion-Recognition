from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib

from emotion_detector import get_model


def _resolve_colormap(name: str):
    """Matplotlib 3.9+ removed cm.get_cmap; use colormaps registry when available."""
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]
    from matplotlib import cm

    return cm.get_cmap(name)

# Cached grad models: (model id, conv layer name, input shape) -> tf.keras.Model
_gradcam_models: dict[tuple[int, str, tuple[int, ...]], tf.keras.Model] = {}


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model; Grad-CAM unavailable.")


def _build_grad_model(
    model: tf.keras.Model,
    conv_layer_name: str,
    input_shape: tuple[int, ...],
) -> tf.keras.Model:
    """
    Rebuild a functional sub-model from an explicit Input.

    Keras 3 Sequential models loaded from .h5 do not expose model.input / model.output
    reliably; tracing layer-by-layer avoids "never been called" errors.
    """
    inp = tf.keras.Input(shape=input_shape)
    x = inp
    conv_out = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == conv_layer_name:
            conv_out = x

    if conv_out is None:
        raise ValueError(f"Conv layer '{conv_layer_name}' not found in model.")

    return tf.keras.Model(inputs=inp, outputs=[conv_out, x])


def _get_grad_model(
    model: tf.keras.Model,
    conv_layer_name: str,
    model_input: np.ndarray,
) -> tf.keras.Model:
    input_shape = tuple(int(s) for s in model_input.shape[1:])
    cache_key = (id(model), conv_layer_name, input_shape)
    if cache_key in _gradcam_models:
        return _gradcam_models[cache_key]

    grad_model = _build_grad_model(model, conv_layer_name, input_shape)
    _gradcam_models[cache_key] = grad_model
    return grad_model


def compute_gradcam_heatmap(
    model_input: np.ndarray,
    class_index: int | None = None,
    conv_layer_name: str | None = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single image.

    Args:
        model_input: Numpy array shaped (1, H, W, C) normalized to [0,1]
        class_index: Optional class index. If None, uses argmax prediction.
        conv_layer_name: Optional conv layer name. If None, uses last Conv2D.

    Returns:
        heatmap: float array in [0, 1] shaped (h, w) (conv feature map size).
    """
    model = get_model()
    if conv_layer_name is None:
        conv_layer_name = _find_last_conv_layer(model)

    grad_model = _get_grad_model(model, conv_layer_name, model_input)

    x = tf.convert_to_tensor(model_input, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]).numpy())
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, conv_out)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients are None; check model and input shape.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    maxv = tf.reduce_max(heatmap)
    heatmap = tf.where(maxv > 0, heatmap / maxv, heatmap)
    return heatmap.numpy().astype(np.float32)


def overlay_heatmap_on_image(
    base_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlay a heatmap onto a PIL image.

    Args:
        base_image: Original image (any mode). Will be converted to RGB.
        heatmap: 2D heatmap in [0,1].
        alpha: blend amount (0..1) for heatmap.
        colormap: matplotlib colormap name.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    img = base_image.convert("RGB")
    w, h = img.size

    hm = np.clip(heatmap, 0.0, 1.0)
    hm_img = Image.fromarray(np.uint8(hm * 255), mode="L").resize((w, h), Image.BICUBIC)
    hm_arr = np.asarray(hm_img, dtype=np.float32) / 255.0

    cmap = _resolve_colormap(colormap)
    colored = (cmap(hm_arr)[:, :, :3] * 255).astype(np.uint8)  # drop alpha channel
    colored_img = Image.fromarray(colored, mode="RGB")

    return Image.blend(img, colored_img, alpha)
