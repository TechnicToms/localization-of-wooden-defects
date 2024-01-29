import torch
from torch.utils.data import DataLoader
import torchvision

import os
import json
import tqdm
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, roc_auc_score

from models.PDN import EfficientAdModelSize
from models.EfficientAD import EfficientAdModule

from datasets.NaturalOak import NaturalOakData, DatasetSplit
from datasets.ImageFolder import InfiniteDataloader, ImageFolderWithoutTarget

from helpers.terminalColor import terminalColor as tc
from metrics.RDR import RDR


DATA_ROOT: str = "/home/tsa/data/NaturalOak/"
DATA_ROOT_IMGNET: str = "/home/tsa/data/imagenet/ILSVRC/Data/CLS-LOC/train"
IMG_SIZE: int = 512
OUT_CHANNELS: int = 384
DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(num_iters: int, datasetName: str="NaturalOak", saveFolder: str="results/EfficientAD/", teacherWeightsPath: str="checkpoints/teacher/teacher_medium_final_state.pth"):
    statePath = os.path.join(saveFolder, datasetName, "stateDicts")
    lossPath = os.path.join(saveFolder, datasetName, "loss")
    
    if not os.path.exists(statePath):
        os.makedirs(statePath)
        print(tc.success + f"Gotcha! Created SavePath: {statePath}")
    if not os.path.exists(lossPath):
        os.makedirs(lossPath)
        print(tc.success + f"Gotcha! Created SavePath: {lossPath}")
        
    #########################
    #   Create Datasets     #
    #########################
    dataOakTrain = NaturalOakData(DATA_ROOT, split=DatasetSplit.TRAIN, prior=False, resize=IMG_SIZE)
    loaderTrain = InfiniteDataloader(DataLoader(dataOakTrain, batch_size=1, shuffle=True, num_workers=6))

    penalty_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((2 * IMG_SIZE, 2 * IMG_SIZE)),
            torchvision.transforms.RandomGrayscale(0.3),
            torchvision.transforms.CenterCrop(IMG_SIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataImageNet = ImageFolderWithoutTarget(DATA_ROOT_IMGNET, transform=penalty_transform)
    loaderPenalty = InfiniteDataloader(DataLoader(dataImageNet, batch_size=1, shuffle=True, num_workers=6))
    
    #####################
    #   Create Models   #
    #####################
    model: torch.nn.Module = EfficientAdModule(OUT_CHANNELS, input_size=(IMG_SIZE, IMG_SIZE), modelSize=EfficientAdModelSize.M, padding=False)
    state_dict = torch.load(teacherWeightsPath, map_location='cpu')
    model.teacher.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model = model.train()
    
    #########################
    #   Optim & Train loop  #
    #########################
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    loss_iteration = torch.zeros(num_iters)
    for i in range(0, num_iters):
        image, _ = next(loaderTrain)
        imagePen: torch.Tensor = next(loaderPenalty)
    
        batchStA = image.to(DEVICE)
        batchPen = imagePen.to(DEVICE)

        optim.zero_grad()

        loss_st, loss_ae, loss_stae = model(batchStA, batchPen)
        
        loss: torch.Tensor = loss_st + loss_ae + loss_stae
        
        loss_iteration[i] = loss.item()
        
        loss.backward()
        optim.step()
        
        if i % 100 == 0: 
            if i < 1:
                print(tc.train + f"Iteration {i}:\t Loss= {loss_iteration[i]}")
            else:
                print(tc.train + f"Iteration {i}:\t Loss= {torch.mean(loss_iteration[i-100:i])}")
                
            torch.save(model.student.state_dict(), os.path.join(statePath, "student.pth"))
            torch.save(model.ae.state_dict(), os.path.join(statePath, "autoencoder.pth"))
            torch.save(model.teacher.state_dict(), os.path.join(statePath, "teacher.pth"))    
            
    torch.save(loss_iteration, os.path.join(lossPath, "loss.pt"))    

def test(dataset:str="NaturalOak", saveFolder: str="results/EfficientAD/"):
    dataOakTest = NaturalOakData(DATA_ROOT, split=DatasetSplit.TEST, prior=False, resize=IMG_SIZE)
    loaderTest = DataLoader(dataOakTest, batch_size=1, shuffle=False)
    
    savePathMaps = os.path.join(saveFolder, f"{dataset}/anomalyMaps")
    if not os.path.exists(savePathMaps):
        os.makedirs(savePathMaps)
        print(tc.success + f"Gotcha! Created path: {savePathMaps}")
    
    #####################
    #   Create Models   #
    #####################
    model: torch.nn.Module = EfficientAdModule(OUT_CHANNELS, input_size=(IMG_SIZE, IMG_SIZE), modelSize=EfficientAdModelSize.M, padding=False)
    state_dict_teacher = torch.load(os.path.join(saveFolder, f"{dataset}/stateDicts/teacher.pth"), map_location='cpu')
    state_dict_student = torch.load(os.path.join(saveFolder, f"{dataset}/stateDicts/student.pth"))
    state_dict_ae = torch.load(os.path.join(saveFolder, f"{dataset}/stateDicts/autoencoder.pth"))
    model.teacher.load_state_dict(state_dict_teacher)
    model.student.load_state_dict(state_dict_student)
    model.ae.load_state_dict(state_dict_ae)
    
    model = model.to(DEVICE)
    model = model.eval()
    
    ######################
    #   prediction loop  #
    ######################
    anomalyMaps = torch.zeros(len(loaderTest), IMG_SIZE, IMG_SIZE)
    groundTruths = torch.zeros(len(loaderTest), IMG_SIZE, IMG_SIZE)
    y_true = torch.zeros(len(loaderTest))
    y_pred = torch.zeros(len(loaderTest))
    for bi, (img, gt) in tqdm.tqdm(enumerate(loaderTest), desc=tc.info + "Computing Predictions", total=len(loaderTest)):
        batch: torch.Tensor = img.to(DEVICE)
        groundTruths[bi, ...] = gt[0, ...]
        y_true[bi] = 1.0 if torch.sum(gt) > 42 else 0.0
        
        with torch.no_grad():
            predictionDict: dict = model(batch)
            
        anomalyMap: torch.Tensor = predictionDict["anomaly_map"]
        anomalyMaps[bi, ...] = anomalyMap
        
        map_st: torch.Tensor = predictionDict["map_st"]
        map_stae: torch.Tensor = predictionDict["map_ae"]
        
        y_pred[bi] = torch.max(anomalyMap)
        
        pngTensor = anomalyMap[0, ...].cpu() * 255.0
        torchvision.io.write_png(pngTensor.type(torch.uint8), os.path.join(savePathMaps, f"{bi}.png"))
    
    #############################
    #   Compute Binary Metrics  #
    #############################
    results_dict = {}
    
    print(tc.info + "Computing ROC Curve for binary classification")
    fpr, tpr, threshold = roc_curve(y_true.type(torch.int), y_pred)
    binaryROC = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve NaturalOak (binary)")
    plt.savefig(os.path.join(saveFolder, dataset, "roc_binary.png"))
    plt.close()
    
    optimal_idx = torch.argmax(torch.sqrt(1 - torch.tensor(fpr)**2 + torch.tensor(tpr)**2))
    optimal_threshold = threshold[optimal_idx] 
    
    auc = roc_auc_score(y_true.type(torch.int), y_pred)
    results_dict["binaryAUC"] = auc
    
    cm = compute_confusion_matrix(torch.where(y_pred > optimal_threshold, 1.0, 0.0), y_true)
    
    tp = cm[0, 0]
    tn = cm[1, 1]
    fn = cm[0, 1]
    fp = cm[1, 0]
    
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    results_dict["accuracy"] = float(accuracy)
    results_dict["precision"] = float(precision)
    results_dict["recall"] = float(recall)
    
    results_dict["confusion_matrix"] = cm.tolist()
    
    ##############################
    #   Computing pixel Metrics  #
    ##############################    
    PixelMetric = RDR()
    
    meanMaps = torch.mean(anomalyMaps, dim=(0, 1, 2))
    stdMaps = torch.std(anomalyMaps, dim=(0, 1, 2))
    
    print(tc.info + "Computing ROC Curve for pixel maps")
    fpr, tpr, threshold = roc_curve(groundTruths.flatten().type(torch.int), anomalyMaps.flatten())
    pixelROC = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve NaturalOak (pixel)")
    plt.savefig(os.path.join(saveFolder, dataset, "roc_pixel.png"))
    plt.close()
    
    pixelAUC = roc_auc_score(groundTruths.flatten().type(torch.int), anomalyMaps.flatten())
    results_dict["pixelAUC"] = pixelAUC
    
    for it in tqdm.tqdm(range(len(loaderTest)), desc=tc.info + "Computing Metrics"):
        cAno = anomalyMaps[it, ...]
        cGt = groundTruths[it, ...]
        
        defectMap = torch.where(cAno > meanMaps + 1.0 * stdMaps, 1.0, 0.0)
        # if y_pred[it] < optimal_threshold:
        #     defectMap = torch.zeros(IMG_SIZE, IMG_SIZE)
        
        if y_true[it] == 1.0:
            PixelMetric.update(defectMap, cGt)
            
    print(tc.success + f"RDR: {PixelMetric.compute()}")
    results_dict["rdr"] = float(PixelMetric.compute())

    jsonObject = json.dumps(results_dict, indent=4)
    with open(os.path.join(saveFolder, dataset, "metrics.json"), "w") as f:
        f.write(jsonObject)
        
    jsonObjectBin = json.dumps(binaryROC, indent=4)
    with open(os.path.join(saveFolder, dataset, "binaryROC.json"), "w") as f:
        f.write(jsonObjectBin)
        
    jsonObjectPix = json.dumps(pixelROC, indent=4)
    with open(os.path.join(saveFolder, dataset, "pixelROC.json"), "w") as f:
        f.write(jsonObjectPix)

def compute_confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    n_data = len(y_true)
    
    cm = torch.zeros((2, 2))
    
    for i in range(n_data):
        yp = y_pred[i]
        yt = y_true[i]
    
        cm[yt.long(), yp.long()] += 1
    
    return cm


if __name__ == '__main__':
    train(10000, "NaturalOak-Medium")
    test("NaturalOak-Medium")
    print(tc.success + 'finished!')