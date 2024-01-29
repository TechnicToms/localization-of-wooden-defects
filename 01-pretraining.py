import torch
from torch.utils.data import DataLoader
import torchvision

from models.featureExtractors import FeatureExtractor
from models.PDN import PDNMedium, PDNSmall

from datasets.ImageFolder import ImageFolderWithoutTarget, InfiniteDataloader

import tqdm

import os

BATCH_SIZE: int = 5
DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_ROOT: str = "/home/tsa/data/imagenet/ILSVRC/Data/CLS-LOC/train"
OUT_CHANNELS: int = 384


grayscale_transform = torchvision.transforms.RandomGrayscale(0.1)  # apply same to both

extractor_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 512)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

pdn_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_transform(image):
    image = grayscale_transform(image)
    return extractor_transform(image), pdn_transform(image)

def train():
    save_path = "checkpoints/teacher"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    backbone = torchvision.models.wide_resnet101_2(
        weights=torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V1)

    extractor = FeatureExtractor(backbone=backbone,
                                 layers_to_extract_from=['layer2', 'layer3'],
                                 device=DEVICE,
                                 input_shape=(3, 512, 512),
                                 out_channels=OUT_CHANNELS)
    model_size = "medium"
    pdn = PDNMedium(OUT_CHANNELS, padding=True)


    train_set = ImageFolderWithoutTarget(DATA_ROOT,
                                         transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True,
                              num_workers=7, pin_memory=True)
    train_loader = InfiniteDataloader(train_loader)

    channel_mean, channel_std = feature_normalization(extractor=extractor,
                                                      train_loader=train_loader)

    pdn.train()
    pdn = pdn.cuda()

    optimizer = torch.optim.Adam(pdn.parameters(), lr=1e-4, weight_decay=1e-5)

    tqdm_obj = tqdm.tqdm(range(60000))
    for iteration, (image_fe, image_pdn) in zip(tqdm_obj, train_loader):

        image_fe = image_fe.cuda()
        image_pdn = image_pdn.cuda()
        
        target = extractor.embed(image_fe)
        target = (target - channel_mean) / channel_std
        prediction = pdn(image_pdn)
        loss = torch.mean((target - prediction)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm_obj.set_description(f'{(loss.item())}')

        if iteration % 10000 == 0:
            torch.save(pdn, os.path.join(save_path, f'teacher_{model_size}_tmp.pth'))
            torch.save(pdn.state_dict(), os.path.join(save_path, f'teacher_{model_size}_tmp_state.pth'))
            
    torch.save(pdn, os.path.join(save_path, f'teacher_{model_size}_final.pth'))
    torch.save(pdn.state_dict(), os.path.join(save_path, f'teacher_{model_size}_final_state.pth'))

@torch.no_grad()
def feature_normalization(extractor, train_loader, steps=10000):

    mean_outputs = []
    normalization_count = 0
    with tqdm.tqdm(desc='Computing mean of features', total=steps) as pbar:
        for image_fe, _ in train_loader:

            image_fe = image_fe.cuda()
            
            output = extractor.embed(image_fe)
            mean_output = torch.mean(output, dim=[0, 2, 3])
            mean_outputs.append(mean_output)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    normalization_count = 0
    with tqdm.tqdm(desc='Computing variance of features', total=steps) as pbar:
        for image_fe, _ in train_loader:
            
            image_fe = image_fe.cuda()
            
            output = extractor.embed(image_fe)
            distance = (output - channel_mean) ** 2
            mean_distance = torch.mean(distance, dim=[0, 2, 3])
            mean_distances.append(mean_distance)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    train()
    print('finished!')