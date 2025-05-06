import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use('Agg')

def visualize_attention(model, image_tensor):
    model.eval()

    if image_tensor.dim() == 5:
        image_tensor = image_tensor.squeeze(0)  

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        print("Input to attention map:", image_tensor.shape)
        att_map = model.get_attention_map(image_tensor) 

    print("Raw attention output shape:", np.shape(att_map))

    att_map_resized = cv2.resize(att_map, (224, 224))
    att_norm = (att_map_resized - att_map_resized.min()) / (att_map_resized.max() - att_map_resized.min())

    threshold = 0.4
    att_masked = np.where(att_norm >= threshold, att_norm, 0)

    selected_img = image_tensor[0].cpu() 
    selected_img_np = selected_img.permute(1, 2, 0).numpy()
    selected_img_np = (selected_img_np - selected_img_np.min()) / (selected_img_np.max() - selected_img_np.min())
    img_rgb = (selected_img_np * 255).astype(np.uint8)

    heatmap = cv2.normalize(att_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img_rgb, 0.6, 0)

    return superimposed_img


def visualize_gradcam(model, image_tensor, target_class):
    model.eval()
    conv_layers = [layer for layer in model.feature_extractor.modules() if isinstance(layer, torch.nn.Conv2d)]
    if len(conv_layers) == 0:
        print("No Conv2D layers found")
        target_layer = None
    else:
        target_layer = conv_layers[-1]  
        print(f"Using target layer for Grad-CAM: {target_layer}")

    if image_tensor.dim() == 3:  
        image_tensor = image_tensor.unsqueeze(0)  
        print(f"3d tensor with added bag dimension: {image_tensor.shape}")

    if image_tensor.dim() == 4:  
        image_tensor = image_tensor.unsqueeze(1)  
        print(f"4d tensor with added bag dimension: {image_tensor.shape}")

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

    # get first image from bag
    if image_tensor.dim() == 5:  
        vis_tensor = image_tensor[:, 0, :, :, :] 
        grad_cam_input = vis_tensor  
    else:
        vis_tensor = image_tensor 
        grad_cam_input = vis_tensor


    try:
        cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
        cam.batch_size = 1 
        grayscale_cam = cam(input_tensor=grad_cam_input, targets=[ClassifierOutputTarget(target_class)])[0]
        print("Grad-CAM was successful!")
    except Exception as e:
        print(f" Grad-CAM Error: {e}")
        import traceback
        traceback.print_exc()  
        return

    img_np = vis_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize 
    img_rgb = (img_np * 255).astype(np.uint8)

    heatmap = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(heatmap, 0.4, img_rgb, 0.6, 0)

    return superimposed_img


def visualize_gradcam_plus(model, image_tensor, target_class):
    model.eval()
    
    conv_layers = [layer for layer in model.feature_extractor.modules() if isinstance(layer, torch.nn.Conv2d)]
    if not conv_layers:
        print("No Conv2D layers.")
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
        print("Grad-CAM++ was successful!")
    except Exception as e:
        print(f"Grad-CAM++ Error: {e}")
        return

    img_np = vis_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  
    img_rgb = (img_np * 255).astype(np.uint8)

    heatmap = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(heatmap, 0.4, img_rgb, 0.6, 0)

    return superimposed_img