import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms as tr
from torchvision.models import vit_h_14
import torchvision.transforms as transforms
import torch.nn.functional as F


class cosineSimilarity:
    """Class tasked with comparing similarity between two images """
    
    def __init__(self, device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.get_model()
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

    def compute_scores(self, support_imgs, target_imgs):
        """Computes cosine similarity between two vectors."""
        # emb_one, emb_two = self.get_embeddings()
        model = self.model 
        support_imgs, target_imgs = support_imgs.cuda(), target_imgs.cuda()
        n_shot = support_imgs.shape[1]
        batch_size = support_imgs.shape[0]

        target_imgs = F.interpolate(target_imgs, size=(518, 518), mode='bilinear', align_corners=False)
        # support_imgs = F.interpolate(support_imgs, size=(518, 518), mode='nearest')
        
        support_reshaped = support_imgs.view(-1, support_imgs.shape[-3], support_imgs.shape[-2], support_imgs.shape[-1])
        support_reshaped = F.interpolate(support_reshaped, size=(518, 518), mode='bilinear', align_corners=False)
        support_imgs = support_reshaped.view(batch_size, n_shot, 3, support_reshaped.shape[-2], support_reshaped.shape[-1])

        results = []
        for i in range(batch_size):
            support_batch = support_imgs[i]
            target = target_imgs[i]
            target = target.unsqueeze(0)
            emb_two = model(target).detach().cpu()

            for j in range(n_shot):
                supp_img = support_batch[j]
                supp_img = supp_img.unsqueeze(0)
                emb_one = model(supp_img).detach().cpu()
                scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
                results.append(scores.item())

        return torch.tensor(results, dtype=torch.float64)



# https://onyekaokonji.medium.com/cosine-similarity-measuring-similarity-between-multiple-images-f289aaf40c2b