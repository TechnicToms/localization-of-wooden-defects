import torch
from torch.utils.data import DataLoader

import os
import json
import tqdm
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, roc_auc_score

from datasets.NaturalOak import NaturalOakData, DatasetSplit

from helpers.terminalColor import terminalColor as tc
from metrics.RDR import RDR


DATA_ROOT: str = "/home/tsa/data/NaturalOak/"
IMG_SIZE: int = 512
DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"

mean_prior: float = 0.0445
std_prior: float = 0.0410


def compute_confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    n_data = len(y_true)
    
    cm = torch.zeros((2, 2))
    
    for i in range(n_data):
        yp = y_pred[i]
        yt = y_true[i]
    
        cm[yt.long(), yp.long()] += 1
    
    return cm

def evaluate(saveFolder: str, dataset: str):
    dataOakTest = NaturalOakData(DATA_ROOT, split=DatasetSplit.TEST, prior=True, resize=IMG_SIZE)
    loaderTest = DataLoader(dataOakTest, batch_size=1, shuffle=False)
    
    metricsSavePath = os.path.join(saveFolder, dataset)
    
    if not os.path.exists(metricsSavePath):
        os.makedirs(metricsSavePath)
        print(tc.success + f"Gotcha! Created Path {metricsSavePath}")
    
    ###############################
    #   Predict and extract maps  #
    ###############################
    AllPriors = torch.zeros(len(loaderTest), IMG_SIZE, IMG_SIZE)
    AllGroundTruths = torch.zeros(len(loaderTest), IMG_SIZE, IMG_SIZE)
    
    y_true = torch.zeros(len(loaderTest), dtype=torch.int)
    y_pred = torch.zeros(len(loaderTest), dtype=torch.float)
    
    for imgID, (img, gt) in tqdm.tqdm(enumerate(loaderTest), desc=tc.info + "Computing pixel metrics"):
        priorNormalized: torch.Tensor = img[0, 2, ...]
        
        prior = (priorNormalized * std_prior) + mean_prior
        AllPriors[imgID] = prior
        AllGroundTruths[imgID] = gt[0, ...]
        
        y_true[imgID] = 1.0 if torch.sum(gt) > 42.0 else 0.0
        y_pred[imgID] = torch.max(prior)

    
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
    plt.savefig(os.path.join(metricsSavePath, "roc_binary.png"))
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
    
    meanMaps = torch.mean(AllPriors, dim=(0, 1, 2))
    stdMaps = torch.std(AllPriors, dim=(0, 1, 2))
    
    print(tc.info + "Computing ROC Curve for pixel maps")
    fpr, tpr, threshold = roc_curve(AllGroundTruths.flatten().type(torch.int), AllPriors.flatten())
    pixelROC = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve NaturalOak (pixel)")
    plt.savefig(os.path.join(metricsSavePath, "roc_pixel.png"))
    plt.close()
    
    pixelAUC = roc_auc_score(AllGroundTruths.flatten().type(torch.int), AllPriors.flatten())
    results_dict["pixelAUC"] = pixelAUC
    
    for it in tqdm.tqdm(range(len(loaderTest)), desc=tc.info + "Computing Metrics"):
        cAno = AllPriors[it, ...]
        cGt = AllGroundTruths[it, ...]
        
        defectMap = torch.where(cAno > meanMaps + 2.70 * stdMaps, 1.0, 0.0)

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


if __name__ == '__main__':
    evaluate(saveFolder="results/Prior/", dataset="NaturalOak")
    print(tc.success + 'finished!')