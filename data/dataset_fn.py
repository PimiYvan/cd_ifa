r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.fss import DatasetFSS
from data.deepglobe import DatasetDeepglobe
from data.isic import DatasetISIC
from data.lung import DatasetLung

from data.deepglobe_IFA import DatasetDeepglobeIFA
from data.isic_IFA import DatasetISICIFA
from data.lung_IFA import DatasetLungIFA
from data.fss_IFA import DatasetFSSIFA

from data.deepglobe_dist import DatasetDeepglobeDist
from data.deepglobe_dist_2 import DatasetDeepglobeDist2
from data.fss_dist import DatasetFSSDist
from data.isic_dist import DatasetISICDist
from data.lung_dist import DatasetLungDist


def custom_collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    return batch

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'fss': DatasetFSS,
            'deepglobe': DatasetDeepglobe,
            'isic': DatasetISIC,
            'lung': DatasetLung,
            'deepglobeifa': DatasetDeepglobeIFA,
            'isicifa': DatasetISICIFA,
            'lungifa': DatasetLungIFA,
            'fssifa': DatasetFSSIFA,
            'deepglobedist': DatasetDeepglobeDist,
            'deepglobedist2': DatasetDeepglobeDist2,
            'fssdist': DatasetFSSDist,
            'isicdist': DatasetISICDist,
            'lungdist': DatasetLungDist,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, collate_fn=custom_collate_fn)

        return dataloader
