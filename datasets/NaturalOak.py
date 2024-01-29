import torch
from torch.utils.data import Dataset
import torchvision

from defecty.core import terminalColor as tc
from enum import Enum
import os


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class NaturalOakData(Dataset):
    MEAN = [0.4378, 0.2408, 0.1030]
    STD = [0.1626, 0.0876, 0.0508]
    
    PRIOR_MEAN = [0.4378, 0.2408, 0.0445]
    PRIOR_STD = [0.1626, 0.0876, 0.0410]
    
    def __init__(self, root: str, split: DatasetSplit=DatasetSplit.TRAIN, prior: bool=False, resize: int=1024) -> None:
        """NaturalOak Dataset

        Args:
            root (str): Path to dataset structure
            split (DatasetSplit, optional): Wether to load training or testing images. Defaults to DatasetSplit.TRAIN.
            prior (bool, optional): If True uses a prior information. Defaults to False.
            resize (int, optional): Side length to resize to. Defaults to 1024.
        """
        super().__init__()
        self.root = root
        self.split = split
        self.prior = prior
        
        self.rgb_per_class, self.data_paths = self.getImageData()
        
        self.rgb_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize, antialias=True),
            torchvision.transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
        if self.prior:
            self.rgb_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize, antialias=True),
                torchvision.transforms.Normalize(mean=self.PRIOR_MEAN, std=self.PRIOR_STD)
            ])
        
        self.mask_transform = torchvision.transforms.Resize(resize, antialias=True)
        
        self.imageSize = (3, resize, resize)
        
    def __repr__(self) -> str:
        return f"NaturalOakData(#{self.__len__()}, split={self.split.value})"
    
    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index):
        anomaly, image_path, prior_path, mask_path = self.data_paths[index]
        
        image = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.RGB) / 255.0
        if self.prior:
            prior_image = torchvision.io.read_image(prior_path, torchvision.io.ImageReadMode.GRAY) / 255.0
            # Replace blue channel with prior
            image[2, ...] = prior_image
            
        image = self.rgb_transform(image)
        
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = torchvision.io.read_image(mask_path, torchvision.io.ImageReadMode.GRAY) / 255.0
            mask = self.mask_transform(mask)
        else:
            mask = torch.zeros((1, self.imageSize[1], self.imageSize[2]))
        
        return image, mask

    def getImageData(self):
        rgb_per_class = {}
        mask_per_class = {}
        prior_per_class = {}
        
        # Get Root directories
        rgb_root_path = os.path.join(self.root, self.split.value)
        mask_root_path = os.path.join(self.root, "ground_truth")
        
        anomaly_types = os.listdir(rgb_root_path)
        
        # Iterate over all types of images
        # - in 'train' there is only a "good" class
        for anomalyName in anomaly_types:
            anomaly_root_path = os.path.join(rgb_root_path, anomalyName)
            anomaly_files = sorted(os.listdir(anomaly_root_path))
            if 'prior' in anomaly_files:
                anomaly_files.remove('prior')
            
            # Add RGB file paths
            rgb_per_class[anomalyName] = [os.path.join(anomaly_root_path, file) for file in anomaly_files]
            
            # Add Mask file paths
            if self.split == DatasetSplit.TEST and anomalyName != "good":
                mask_files = sorted(os.listdir(os.path.join(mask_root_path, anomalyName)))
                mask_per_class[anomalyName] = [os.path.join(mask_root_path, anomalyName, file) for file in mask_files]
                
            # Add prior file paths
            prior_per_class[anomalyName] = [os.path.join(anomaly_root_path, 'prior', file) for file in anomaly_files]
            
        # Unroll
        data_paths = []
        for anomalyName in sorted(rgb_per_class.keys()):
            for k, rgb_path in enumerate(rgb_per_class[anomalyName]):
                data_point = [anomalyName, rgb_path, prior_per_class[anomalyName][k]]
                if self.split == DatasetSplit.TEST and anomalyName != "good":
                    data_point.append(mask_per_class[anomalyName][k])
                else:
                    data_point.append(None)
                    
                data_paths.append(data_point)
                
        return rgb_per_class, data_paths
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    
    path = "checkpoints/deleteMe/wood/"
    data = NaturalOakData(path, split=DatasetSplit.TEST, prior=True)
    
    img, gt = data[0]
    
    print(tc.success + 'finished!')