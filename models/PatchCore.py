import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import tqdm
import os

from helpers.terminalColor import terminalColor as tc

from sampling.greedyKCenter import kCenterGreedy
from sampling.greedyKCenterTorch import ApproximateGreedyCoresetSampler
from sklearn.random_projection import SparseRandomProjection
import faiss


class PatchCore(torch.nn.Module):
    def __init__(self, n_neighbors: int, faissPath: str="checkpoints/faiss/") -> None:
        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Feature extraction related stuff (Hook + Network)
        self.features: list[torch.Tensor] = []
        def hook(module, input, output) -> None:
            self.features.append(output)
        
        # Construct model and add hooks 
        self.model: torch.nn.Module = torchvision.models.wide_resnet101_2(weights=torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
        self.model.to(self.device)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        
        self.model.eval()
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False   
        
        self.memory_bank = []    
        self.coreset_sampling_ratio = 0.01
        self.n_neighbors = n_neighbors
        
        self.faissPath: str = faissPath
        if not os.path.exists(self.faissPath):
            os.makedirs(self.faissPath)
            print(tc.success + f"Gotcha! Created path {self.faissPath}")
        
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        self.features = []
        _ = self.model(x)
        return self.features

    def fit(self, loader: DataLoader):
        """Fit's the model (generates the MemoryBank with coreset subsampling).

        Args:
            loader (DataLoader): Dataloader of images
        """
        # Create Memory bank
        self.__sampling(loader)
        
        # Coreset subsampling
        self.__coreset_subsampling()
    
    def predict(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # check if fit was called before!
        # if not hasattr(self, "index"):
        self.__load_index()
        
        # Get params of image and resize
        w_img = batch.shape[-1]
        resizeAndBlurLayer = torchvision.transforms.Compose([
            torchvision.transforms.Resize(w_img, antialias=True),
            torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1.4)        
        ])

        # Patch extraction and aggregation
        features = self(batch.to(self.device))
        embeddings = self.__aggregation(features)
        
        embedding_ = self.embedding_concat(embeddings[0], embeddings[1])
        
        embedding_test = np.array(self.reshape_embedding(np.array(embedding_)))
        score_patches, _ = self.index.search(embedding_test , k=self.n_neighbors)
        anomaly_map = score_patches[:,0].reshape((64, 64))
        N_b = score_patches[np.argmax(score_patches[:,0])]
        # w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
        
        # Image-level score
        score = torch.max(torch.tensor(score_patches[:,0]))# w * max(score_patches[:,0]) 
        
        # Scale and blur Anomalymap
        anomaly_map_resized_blur = resizeAndBlurLayer(torch.tensor(anomaly_map)[None, None, ...])
        
        return score, anomaly_map_resized_blur
    
    def __load_index(self):
        self.index = faiss.read_index(os.path.join(self.faissPath, 'index.faiss'))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
    
    def __coreset_subsampling(self) -> list:
        # Conversion to numpy array
        total_embeddings = torch.vstack(self.memory_bank)
        
        selector = ApproximateGreedyCoresetSampler(self.coreset_sampling_ratio, 
                                                    device="cpu")
        n_datapoints = total_embeddings.shape[0]
        selected_idx = torch.randint(0, n_datapoints, (int(n_datapoints / 5),))
        self.memory_bank_coreset = selector.run(total_embeddings[selected_idx, ...])
        
        #faiss
        self.index = faiss.IndexFlatL2(self.memory_bank_coreset.shape[1])
        self.index.add(self.memory_bank_coreset.numpy()) 
        faiss.write_index(self.index,  os.path.join(self.faissPath, 'index.faiss'))
        
    def __sampling(self, loader: DataLoader) -> None:
        """Samples features from images given by the ``loader`` and stores them into the MemoryBank.

        Args:
            loader (DataLoader): Dataloader, that returns in format (img, gt)
        """
        for it, (img, gt) in tqdm.tqdm(enumerate(loader), desc=tc.train + "Constructing memory bank", total=len(loader)):
            features = self(img.to(self.device))

            embeddings = self.__aggregation(features)
            embedding = self.embedding_concat(embeddings[0], embeddings[1])

            self.memory_bank.extend(self.reshape_embedding(embedding))
            
    def __aggregation(self, features: torch.Tensor) -> list:
        embeddings = []
        # Loop over all Features and apply Aggregation function
        for feature in features:
            avgPool = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(avgPool(feature))
        return embeddings

    @staticmethod
    def embedding_concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concatenates the embeddings in a right way.
        
        Stolen from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

        Args:
            x (torch.Tensor): Embedding 0
            y (torch.Tensor): Embedding 1

        Returns:
            torch.Tensor: Concatenated embedding
        """
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
        return z

    @staticmethod
    def reshape_embedding(embedding):
        embedding_list = []
        for k in range(embedding.shape[0]):
            for i in range(embedding.shape[2]):
                for j in range(embedding.shape[3]):
                    embedding_list.append(embedding[k, :, i, j])
        return embedding_list


if __name__ == '__main__':
    from datasets.NaturalOak import NaturalOakData, DatasetSplit
    
    train_data = NaturalOakData("/home/tsa/data/05-MV-AD/wood/", split=DatasetSplit.TRAIN, resize=512)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    
    test_data = NaturalOakData("/home/tsa/data/05-MV-AD/wood/", split=DatasetSplit.TEST, resize=512)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    
    model = PatchCore(n_neighbors=12, faissPath="checkpoints/faiss/" + train_data.NAME)
    model.fit(train_loader)
    
    print(tc.info + "Evaluating")
    scores = torch.zeros(len(test_loader))
    for it, (batch, gt) in enumerate(test_loader):
        score, anomaly_map = model.predict(batch[0, None, ...])
        scores[it] = score
    
    print(tc.success + 'finished!')