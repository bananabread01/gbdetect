import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class Attention(nn.Module):
  def __init__(self, num_classes=3):
    super(Attention, self).__init__()
    self.L = 512 
    self.D = 128  
    self.K = 1    

    weights = EfficientNet_B0_Weights.DEFAULT
    efficientnet = efficientnet_b0(weights=weights)

    self.feature_extractor = nn.Sequential(
        efficientnet.features 
    )

    self.pool = nn.AdaptiveAvgPool2d((7, 7)) 

    self.feature_extractor_part2 = nn.Sequential(
       nn.Linear(1280 * 7 * 7, self.L), 
       nn.ReLU(),
       nn.Dropout(0.5),
       nn.Linear(self.L, self.L),
       nn.ReLU(),
       nn.Dropout(0.5)
    )

    self.attention = nn.Sequential(
       nn.Linear(self.L, self.D),
       nn.Tanh(),
       nn.Linear(self.D, self.K)
    )

    self.classifier = nn.Linear(self.L * self.K, num_classes)

  def forward(self, x):
    B, bag_size, C, H, W = x.shape
    x = x.view(B * bag_size, C, H, W)
    features = self.feature_extractor(x)
    features = self.pool(features)
    features = features.view(B * bag_size, -1)
    H_features = self.feature_extractor_part2(features)
    H_features = H_features.view(B, bag_size, -1)

    A = self.attention(H_features.view(B * bag_size, -1))
    A = A.view(B, bag_size, self.K).transpose(1, 2)
    A = F.softmax(A, dim=2)

    M = torch.bmm(A, H_features)  
    M = M.view(B, -1) 
    logits = self.classifier(M)
    probs = torch.softmax(logits, dim=1)
    return logits, probs, A

  def calculate_classification_error(self, X, Y):
    logits, _, _ = self.forward(X)
    preds = torch.argmax(logits, dim=1)
    error = 1.0 - preds.eq(Y).cpu().float().mean().item()
    return error, preds

  def calculate_objective(self, X, Y):
    logits, _, A = self.forward(X)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, Y)
    return loss, A
  
  def get_attention_map(self, x):
    self.eval()
    with torch.no_grad():
      if x.ndimension() == 5:
          x = x[:, 0, :, :, :] 

      print(f"Input shape after adjustment: {x.shape}")

      features = self.feature_extractor(x)        
      features = self.pool(features)            

      att_map = features.sum(dim=1)             
      att_map = att_map[0].cpu().numpy()        
      return att_map 