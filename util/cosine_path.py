import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms as tr
from torchvision.models import vit_h_14, vit_l_16
import torchvision.transforms as transforms
import torch.nn.functional as F


class cosineSimilarity:
    """Class tasked with comparing similarity between two images """
    
    def __init__(self, device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.get_model()
        # self.model = self.get_model_v_16()
        # self.image_path_1 = image_path_1
        # self.image_path_2 = image_path_2
    
    def get_model(self):
        """Instantiates the feature extracting model 
        
        Parameters
        ----------
        model

        Returns
        -------
        Vision Transformer model object

        """
        wt = torchvision.models.ViT_H_14_Weights.DEFAULT
        model = vit_h_14(weights=wt)
        model.heads = nn.Sequential(*list(model.heads.children())[:-1])
        model = model.cuda()
        return model

    def get_model_v_16(self):
        """Instantiates the feature extracting model 
        
        Parameters
        ----------
        model

        Returns
        -------
        Vision Transformer model object

        """
        wt = torchvision.models.ViT_L_16_Weights.DEFAULT
        model = vit_l_16(weights=wt)
        
        # Remove the last classification layer (for transfer learning or feature extraction)
        model.heads = nn.Sequential(*list(model.heads.children())[:-1])
        
        # Move the model to GPU if available
        model = model.cuda() if torch.cuda.is_available() else model
        return model

    def process_test_image(self, image_path):
        """Processing images

        Parameters
        ----------
        image_path :str

        Returns
        -------
        Processed image : str

        """
        img = Image.open(image_path)
        transformations = tr.Compose([tr.ToTensor(),
                                        tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        tr.Resize((518, 518))])
        img = transformations(img).float()
        img = img.unsqueeze_(0)
        
        img = img.to(self.device)

        return img

    def compute_scores(self, support_img, target_img):
        """Computes cosine similarity between two vectors."""
        # emb_one, emb_two = self.get_embeddings()
        img1 = self.process_test_image(support_img)
        img2 = self.process_test_image(target_img)
        model = self.model

        emb_one = model(img1).detach().cpu()
        emb_two = model(img2).detach().cpu()

        scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
        return scores 

# https://onyekaokonji.medium.com/cosine-similarity-measuring-similarity-between-multiple-images-f289aaf40c2b