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


    # def get_attention_map(self, x):
    #   self.eval()
    #   with torch.no_grad():
    #       if x.ndimension() == 5:
    #         x = x.unsqueeze(0)  
    #       elif x.ndimension() == 4:  
    #         pass
    #       else:
    #         raise ValueError(f"Unexpected input shape in get_attention_map: {x.shape}")

    #       print(f"Input shape after adjustment: {x.shape}")
    #       bag_size, C, H, W = x.shape

    #       features = self.feature_extractor(x) 
    #       features = self.pool(features)     
    #       features = self.extra_conv_layers(features)
    #       features = features.view(bag_size, -1)  

    #       H_features = self.feature_extractor_part2(features)  

    #       A = self.attention(H_features)  
    #       A = F.softmax(A, dim=0)  

    #       att_map = A[:, 0].cpu().numpy()  
    #       return att_map
      
    def get_attention_map(self, x):
      self.eval()
      with torch.no_grad():
        if x.ndimension() == 5:
            x = x[:, 0, :, :, :]  # Remove bag dim if needed

        print(f"Input shape after adjustment: {x.shape}")  # [B, C, H, W]

        features = self.feature_extractor(x)  # [B, C, H, W]

        # Optional: apply pooling or extra conv layers
        features = self.pool(features)  # [B, C, H', W']
        features = self.extra_conv_layers(features)  # still [B, C, H', W']

        # Reduce to attention weights per spatial location
        # For example, sum over channels to create heatmap
        att_map = features.sum(dim=1)  # [B, H', W']

        att_map = att_map[0].cpu().numpy()  # Convert first image in batch
        return att_map









    
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


