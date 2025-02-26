# utils.py
import numpy as np
import torch


def generate_gradcam(model, input_tensor, target_class):
    """
    Generates a Grad-CAM heatmap for the given input_tensor and target_class.
    Note: Adjust the target layer depending on your model's architecture.
    """
    gradients = {}
    activations = {}

    def save_activation(module, input, output):
        activations['value'] = output

    def save_gradient(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    # Identify the target layer. Adjust this based on your model's structure.
    #target_layer = model.features[-1]
    target_layer = model.model.layer4
    # Register hooks
    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_backward_hook(save_gradient)

    # Forward pass
    output = model(input_tensor)
    probabilities_tensor = torch.nn.functional.softmax(output, dim=1)[0]
    predicted_class = torch.argmax(probabilities_tensor).item()
    score = probabilities_tensor[predicted_class]

    # If your output is a vector of logits, select the logit corresponding to target_class.
    ##if output.dim() == 2:
      #  score = output[0, target_class]
    #else:
     #   score = output[0]

    #score = output[0, 0]

    # Backward pass: compute gradients with respect to the target score
    model.zero_grad()
    score.backward(retain_graph=True)

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Get the saved activations and gradients
    grads_val = gradients['value'][0].detach().cpu().numpy()    # shape: [channels, h, w]
    activations_val = activations['value'][0].detach().cpu().numpy()  # shape: [channels, h, w]

    # Compute the weights: global average pooling over the gradients
    weights = np.mean(grads_val, axis=(1, 2))  # shape: [channels]

    # Compute the weighted combination of forward activation maps
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations_val[i, :, :]

    # Apply ReLU to the heatmap (only positive contributions)
    cam = np.maximum(cam, 0)

    # Normalize the heatmap to a range [0, 1]
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    return cam
