import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms as tr
from torchvision.models import vit_h_14


class cosineSimilarity:
    """Class tasked with comparing similarity between two images """
    
    def __init__(self, device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.image_path_1 = image_path_1
        # self.image_path_2 = image_path_2
    
    def model(self):
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
        model = model.to(self.device)

        return model

    def get_embeddings(self):
        """Computer embessings given images 
        
        Parameters
        image_paths : str

        Returns
        -------
        embeddings: np.ndarray

        """
        img1 = self.process_test_image(self.image_path_1)
        img2 = self.process_test_image(self.image_path_2)
        model = self.model()

        emb_one = model(img1).detach().cpu()
        emb_two = model(img2).detach().cpu()

        return emb_one, emb_two

    def compute_scores(self, img1, img2):
        """Computes cosine similarity between two vectors."""
        # emb_one, emb_two = self.get_embeddings()
        model = self.model()
        emb_one = model(img1).detach().cpu()
        emb_two = model(img2).detach().cpu()
        scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)

        return scores.numpy().tolist()