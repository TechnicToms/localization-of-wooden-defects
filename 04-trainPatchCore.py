import torch
from torch.utils.data import DataLoader
import torchvision

import os
import json
import tqdm
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, roc_auc_score
from metrics.RDR import RDR

from models.PatchCore import PatchCore

from helpers.terminalColor import terminalColor as tc
from datasets.NaturalOak import NaturalOakData, DatasetSplit


DATA_ROOT: str = "/home/tsa/data/NaturalOak/"
IMG_SIZE: int = 512
DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"


def trainPatchCore(datasetName: str="NaturalOak"):
    train_data = NaturalOakData(DATA_ROOT, split=DatasetSplit.TRAIN, prior=False, resize=IMG_SIZE)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        
    model = PatchCore(n_neighbors=12, faissPath=f"results/PatchCore/{datasetName}/faiss/")
    model.fit(train_loader)
    
    return model

def testPatchCore(model: PatchCore, savePath: str="results/PatchCore/NaturalOak"):
    test_data = NaturalOakData(DATA_ROOT, split=DatasetSplit.TEST, prior=False, resize=IMG_SIZE)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    anoSavePath = os.path.join(savePath, "anomalyMaps")
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        print(tc.success + f"Gotcha! Created Path: {savePath}")
    if not os.path.exists(anoSavePath):
        os.makedirs(anoSavePath)
        print(tc.success + f"Gotcha! Created Path: {anoSavePath}")
            
    #################################
    #   Prediction of anomalyMaps   #
    #################################
    print(tc.info + "Evaluating")
    y_true = torch.zeros(len(test_loader))
    scores = torch.zeros(len(test_loader))
    
    anomalyMaps = torch.zeros(len(test_loader), IMG_SIZE, IMG_SIZE)
    groundTruthMaps = torch.zeros(len(test_loader), IMG_SIZE, IMG_SIZE)
    
    for it, (batch, gt) in enumerate(test_loader):
        score, anomaly_map = model.predict(batch[0, None, ...])
    
        scores[it] = score
        y_true[it] = 1.0 if torch.sum(gt) > 42.0 else 0.0
        anomalyMaps[it, ...] = anomaly_map[0, 0, ...]
        groundTruthMaps[it, ...] = gt[0, 0, ...]
        
        pngTensor = (anomaly_map - torch.min(anomaly_map)) / (torch.max(anomaly_map) - torch.min(anomaly_map)) * 255.0
        pngTensor = pngTensor[0, ...].cpu()
        torchvision.io.write_png(pngTensor.type(torch.uint8), os.path.join(anoSavePath, f"{it}.png"))
        
    #################
    #   Binary ROC  #
    #################
    results_dict = {}
    
    print(tc.info + "Computing ROC Curve for binary classification")
    fpr, tpr, threshold = roc_curve(y_true.type(torch.int), scores)
    
    binaryROC = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve NaturalOak (binary)")
    plt.savefig(os.path.join(savePath, "roc_binary.png"))
    plt.close()
    
    optimal_idx = torch.argmax(torch.sqrt(1 - torch.tensor(fpr)**2 + torch.tensor(tpr)**2))
    optimal_threshold = threshold[optimal_idx] 
    
    auc = roc_auc_score(y_true.type(torch.int), scores)
    results_dict["binaryAUC"] = auc
    
    cm = compute_confusion_matrix(torch.where(scores > optimal_threshold, 1.0, 0.0), y_true)
    
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
    fpr, tpr, threshold = roc_curve(groundTruthMaps.flatten().type(torch.int), anomalyMaps.flatten())
    
    pixelROC = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve NaturalOak (pixel)")
    plt.savefig(os.path.join(savePath, "roc_pixel.png"))
    plt.close()
    
    pixelAUC = roc_auc_score(groundTruthMaps.flatten().type(torch.int), anomalyMaps.flatten())
    results_dict["pixelAUC"] = pixelAUC
    
    for it in tqdm.tqdm(range(len(test_loader)), desc=tc.info + "Computing Metrics"):
        cAno = anomalyMaps[it, ...]
        cGt = groundTruthMaps[it, ...]
        
        defectMap = torch.where(cAno > meanMaps + 2.3 * stdMaps, 1.0, 0.0)
        # if y_pred[it] < optimal_threshold:
        #     defectMap = torch.zeros(IMG_SIZE, IMG_SIZE)
        
        if y_true[it] == 1.0:
            PixelMetric.update(defectMap, cGt)
            
    print(tc.success + f"RDR: {PixelMetric.compute()}")
    results_dict["rdr"] = float(PixelMetric.compute())

    jsonObject = json.dumps(results_dict, indent=4)
    with open(os.path.join(savePath, "metrics.json"), "w") as f:
        f.write(jsonObject)
        
    jsonObjectBin = json.dumps(binaryROC, indent=4)
    with open(os.path.join(savePath, "binaryROC.json"), "w") as f:
        f.write(jsonObjectBin)
        
    jsonObjectPix = json.dumps(pixelROC, indent=4)
    with open(os.path.join(savePath, "pixelROC.json"), "w") as f:
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
    trainedModel = trainPatchCore()
    testPatchCore(trainedModel)
    print(tc.success + 'finished!')