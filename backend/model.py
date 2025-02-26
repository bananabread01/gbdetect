# # model.py (example)
# import torch
# import torch.nn as nn
# import torchvision.models as models

# class GallbladderCancerDetector(nn.Module):
#     def __init__(self):
#         super(GallbladderCancerDetector, self).__init__()
#         self.model = models.resnet18(pretrained=True)  # Load ResNet18
#         self.model.fc = nn.Linear(512, 3)  # Change output layer for 3 classes

#     def forward(self, x):
#         return self.model(x)

import torch
import torch.nn as nn
import torchvision.models as models

class GatedAttentionMILPooling(nn.Module):
    def __init__(self, in_features):
        super(GatedAttentionMILPooling, self).__init__()
        # dense layers
        self.attention_gate = nn.Linear(in_features, in_features)
        self.feature_gate   = nn.Linear(in_features, in_features)
        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def forward(self, x):
        # x: feature map of shape (B, C, H, W)
        B, C, H, W = x.shape
        # Flatten spatial dimensions: shape (B, H*W, C)
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        # calculate attention and gated features for each instance
        attn = self.sigmoid(self.attention_gate(x_flat))  # shape: (B, H*W, C)
        feat = self.tanh(self.feature_gate(x_flat))         # shape: (B, H*W, C)
        weighted = attn * feat  # element-wise multiplication

        # Expand dimension and max pooling for the instances (H*W)
        # weighted: (B, H*W, C) -> (B, 1, H*W, C)
        weighted = weighted.unsqueeze(1)
        pooled, _ = weighted.max(dim=2)  # pooled: (B, 1, C)
        pooled = pooled.squeeze(1)       # pooled: (B, C)
        return pooled

# full model
class MILModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MILModel, self).__init__()
        # load EfficientNet-B0
        base_model = models.efficientnet_b0(pretrained=True) # outputs 1280 channels.
        self.base = base_model.features  # convolutional feature extractor
        # freeze initial layers 
        # for param in list(self.base.parameters())[:100]:
        #     param.requires_grad = False

        self.mil_pooling = GatedAttentionMILPooling(in_features=1280)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        features = self.base(x)   # shape: (B, 1280, H, W)  H=W=7
        pooled = self.mil_pooling(features)  # shape: (B, 1280)
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits

    def get_attention_map(self, x):
        """
        Compute an approximate attention map from the MIL module.
        The method runs the base network and then computes the attention scores
        from the attention gate of the MIL pooling. We then average over channels.
        """
        self.eval()
        with torch.no_grad():
            features = self.base(x)  # (B, 1280, H, W)
            B, C, H, W = features.shape
            x_flat = features.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            # Compute attention scores (using same linear layer as MIL pooling)
            attn_scores = self.mil_pooling.sigmoid(self.mil_pooling.attention_gate(x_flat))  # (B, H*W, C)
            # Average attention over the channel dimension to get a 1-channel map:
            attn_map = attn_scores.mean(dim=2).view(B, H, W)
        return attn_map










    
  #  def __init__(self):
   #     super(GallbladderCancerDetector, self).__init__()

        # Feature extraction layers (example with CNN)
       # self.features = nn.Sequential(
        #    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
         #   nn.ReLU(),
          #  nn.MaxPool2d(kernel_size=2, stride=2),
#
 #           nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
  #          nn.ReLU(),
   #         nn.MaxPool2d(kernel_size=2, stride=2)
    #    )

        # Fully Connected (FC) classifier
     #   self.classifier = nn.Sequential(
      #      nn.Linear(64 * 56 * 56, 128),  # Change this to match 200704, based on feature map size
       #     nn.ReLU(),
        #    nn.Linear(128, 3)  # Output a single value for binary classification
        #)

    
  #  def forward(self, x):
   #     x = self.features(x)
        
        # Debugging: Print the output shape before flattening
    #    print("Feature map shape before flattening:", x.shape)

       # x = x.view(x.size(0), -1)  # Flatten
     #   print("Flattened feature map shape:", x.shape)

       # x = self.classifier(x)
      #  return x


# Instantiate the model
#model = GallbladderCancerDetector()


