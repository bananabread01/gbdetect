import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use('Agg')

def visualize_attention(model, image_tensor):
    model.eval()

    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0] 

    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        att_map = model.get_attention_map(image_tensor)  

    print("Raw attention output type:", type(att_map))
    print("Raw attention output:", att_map)


    att_map = torch.tensor(att_map, device=device)
    print("Attention map shape:", att_map.shape)


# If it's a list or 1D scalar-like value, raise an error early
    if att_map.ndim == 1 and att_map.shape[0] == 1:
        raise ValueError("The attention map returned by the model has shape (1,). It might be invalid or scalar.")

    # If it's 3D (e.g. [N_heads, H, W]), average or select a head
    if att_map.ndim == 3:
        att_map = att_map.mean(dim=0)

# If it's not 2D by now, throw an error
    if att_map.ndim != 2:
        raise ValueError(f"Expected attention map to be 2D after processing, but got shape {att_map.shape}")

    # Continue with plotting
    att_map = att_map.cpu().numpy()

    import matplotlib.pyplot as plt
    plt.imshow(att_map, cmap='hot')
    plt.colorbar()
    plt.title("Raw Attention Map (Before Resize)")
    plt.axis('off')
    plt.savefig("attention_map_ab2_debug.png", bbox_inches='tight', pad_inches=0)
    plt.close()



    # att_map = att_map.mean(dim=0)  
    # att_map = att_map.cpu().numpy()  

    img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() 
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_rgb = (img_np * 255).astype(np.uint8)

    h, w, _ = img_rgb.shape  
    # att_map_resized = cv2.resize(att_map, (w, h))

    att_map_resized = cv2.resize(att_map, (w, h))
    # att_map_resized = (img_np * 255).astype(np.uint8)

    att_norm = (att_map_resized - att_map_resized.min()) / (att_map_resized.max() - att_map_resized.min())

    threshold = 0.7
    att_masked = np.where(att_norm >= threshold, att_norm, 0)


    heatmap = cv2.normalize(att_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend attention map with original image
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img_rgb, 0.6, 0)

    return superimposed_img  # NumPy image: H x W x 3


def visualize_gradcam(model, image_tensor, target_class):
    model.eval()
    conv_layers = [layer for layer in model.feature_extractor.modules() if isinstance(layer, torch.nn.Conv2d)]
    if len(conv_layers) == 0:
        print("No Conv2D layers found! Grad-CAM may not work.")
        target_layer = None
    else:
        target_layer = conv_layers[-1]  
        print(f"Using target layer for Grad-CAM: {target_layer}")

    if image_tensor.dim() == 3:  
        image_tensor = image_tensor.unsqueeze(0)  
        print(f"Modified tensor (3d) to add bag dimension: {image_tensor.shape}")

    if image_tensor.dim() == 4:  
        image_tensor = image_tensor.unsqueeze(1)  
        print(f"Modified tensor (4d) to add bag dimension: {image_tensor.shape}")

    # wrapper function
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            if x.dim() == 4:  
                x = x.unsqueeze(1)  

            logits, _, _ = self.model(x)
            return logits

    wrapped_model = ModelWrapper(model)

    if image_tensor.dim() == 5:  
        vis_tensor = image_tensor[:, 0, :, :, :] 
        grad_cam_input = vis_tensor  
    else:
        vis_tensor = image_tensor 
        grad_cam_input = vis_tensor


    try:
        cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
        cam.batch_size = 1  # Set batch size
        grayscale_cam = cam(input_tensor=grad_cam_input, targets=[ClassifierOutputTarget(target_class)])[0]
        print("Grad-CAM computed successfully!")
    except Exception as e:
        print(f" Grad-CAM Error: {e}")
        import traceback
        traceback.print_exc()  
        return

    img_np = vis_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize to [0, 1]
    img_rgb = (img_np * 255).astype(np.uint8)

    # Resize CAM and apply heatmap
    heatmap = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img_rgb, 0.6, 0)

    return superimposed_img


def visualize_gradcam_plus(model, image_tensor, target_class):
    model.eval()
    
    conv_layers = [layer for layer in model.feature_extractor.modules() if isinstance(layer, torch.nn.Conv2d)]
    if not conv_layers:
        print("No Conv2D layers found! Grad-CAM++ may not work.")
        return
    target_layer = conv_layers[-1] 
    print(f"Using target layer for Grad-CAM++: {target_layer}")

    if image_tensor.dim() == 4:  
        image_tensor = image_tensor.unsqueeze(1)  
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            if x.dim() == 4: 
                x = x.unsqueeze(1) 
            logits, _, _ = self.model(x)
            return logits

    wrapped_model = ModelWrapper(model)

    if image_tensor.dim() == 5:  
        vis_tensor = image_tensor[:, 0, :, :, :] 
    else:
        vis_tensor = image_tensor

    try:
        cam = GradCAMPlusPlus(model=wrapped_model, target_layers=[target_layer])
        cam.batch_size = 1
        grayscale_cam = cam(input_tensor=vis_tensor, targets=[ClassifierOutputTarget(target_class)])[0]
        print("Grad-CAM++ computed successfully!")
    except Exception as e:
        print(f"Grad-CAM++ Error: {e}")
        return

    img_np = vis_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize to [0, 1]
    img_rgb = (img_np * 255).astype(np.uint8)

    # Resize CAM to match image
    heatmap = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on image
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img_rgb, 0.6, 0)

    # ✅ Output is a NumPy array (H, W, 3), same shape as original image
    return superimposed_img