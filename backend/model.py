import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Attention(nn.Module):
    def __init__(self, num_classes=3):
      super(Attention, self).__init__()
      self.L = 512  # Size of the fully-connected layer
      self.D = 256  # Attention layer size
      self.K = 1    # Number of attention heads

      resnet = models.resnet18(pretrained=True)

      # if freeze_backbone:
      #   for param in list(resnet.parameters())[:-2]:
      #     param.requires_grad = False

      self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
      self.pool = nn.AdaptiveAvgPool2d((30, 30))  

      self.extra_conv_layers = nn.Sequential(
          nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), 
          nn.ReLU(),
          nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
          nn.ReLU()
      )

      self.feature_extractor_part2 = nn.Sequential(
          nn.Linear(64 * 30 * 30, self.L),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(self.L, self.L),
          nn.ReLU(),
          nn.Dropout(0.5),
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
      features = self.extra_conv_layers(features)
      features = features.view(B * bag_size, -1)  
      H_features = self.feature_extractor_part2(features)  
      H_features = H_features.view(B, bag_size, -1) 
      A = self.attention(H_features.view(B * bag_size, -1))  
      A = A.view(B, bag_size, self.K).transpose(1, 2)   
      temperature = 0.5  # Adjust 
      A = F.softmax(A / temperature, dim=2)

      M = torch.bmm(A, H_features) 
      M = M.view(B, -1) 
      logits = self.classifier(M)
      probs = F.log_softmax(logits, dim=1).exp()
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
        features = self.extra_conv_layers(features)  

        
        att_map = features.sum(dim=1) 

        att_map = att_map[0].cpu().numpy()  
        return att_map









    