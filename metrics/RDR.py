import torch
from torchmetrics import Metric

from helpers.terminalColor import terminalColor as tc
from helpers.regionfill import SequentialLabeling


class RDR(Metric):
    """Region Detection Rate (RDR) Metric.
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = True
    
    def __init__(self, alpha: float=0.5, theta_p: float=0.2, theta_d: float=0.5) -> None:
        """Region Detection Rate (RDR) metric. 
        
        Implementation of:
        
        T. Huxohl and F. Kummert, "Region Detection Rate: An Applied Measure for Surface Defect Localization," 2021 IEEE International Conference on Signal and Image Processing Applications (ICSIPA), Kuala Terengganu, Malaysia, 2021, pp. 111-116, doi: 10.1109/ICSIPA52582.2021.9576810.

        Args:
            alpha (float, optional): Prediction weighting factor. Defaults to 0.5.
            theta_p (float, optional): Proportion that must cover defect ``i``. Defaults to 0.2.
            theta_d (float, optional): Proportion, that all prediction have to cover for correctly identifying defect ``i``. Defaults to 0.5.
        """
        super().__init__()
        # Write parameters to class
        self.alpha = alpha
        self.theta_p = theta_p
        self.theta_d = theta_d
        
        # Add states
        self.add_state("rdr_sum", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_imgs", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
    
    def __repr__(self):
        return tc.info + f"RDR Metric(alpha={self.alpha}, theta_p={self.theta_p}, theta_d={self.theta_d})"
    
    def _compute_Pi_nPi(self, Di: torch.Tensor, Pj: torch.Tensor, n: int, m: int) -> tuple[dict, dict]:
        """Generates the set of predictions, which contributes (``P_i``) and the set of predictions, 
        that doesn't contribute (``nP_i``) to the localization of a defect ``i``.

        Args:
            Di (torch.Tensor): Set of Pixels belonging to defect ``i``.
            Pj (torch.Tensor): Set of Pixels belonging to prediction ``j``.
            n (int): Number of defects on the ground truth.
            m (int): Number of predictions on the predicted map.

        Returns:
            tuple[dict, dict]: P_i, nP_i
        """
        P_i = {}
        nP_i = {}
        for i in range(0, n):
            # i-th set of pixel in D
            di_ = torch.zeros(Di.shape)
            di_[torch.where(Di == i + 1)] = 1
            
            contributing_predictions = set()
            non_contribution_predictions = set()
            for j in range(0, m):
                # j-th set of pixel in P
                pj_ = torch.zeros(Pj.shape)
                pj_[torch.where(Pj == j + 1)] = 1
                
                intersection_di_pj = torch.zeros(di_.shape)
                intersection_di_pj[torch.where(pj_ + di_ == 2)] = 1
                
                if ((torch.sum(intersection_di_pj) / torch.sum(pj_)) >= self.theta_p):
                    contributing_predictions.add(j)
                else:
                    non_contribution_predictions.add(j)
                    
            P_i[i] = contributing_predictions
            nP_i[i] = non_contribution_predictions
            
        return P_i, nP_i
    
    def _compute_C(self, Di: torch.Tensor, Pj: torch.Tensor, P_i: dict, n: int) -> set:
        """Computes the set of defects, that are treated as being correctly located.

        Args:
            Di (torch.Tensor): Set of Pixels belonging to defect ``i``.
            Pj (torch.Tensor): Set of Pixels belonging to prediction ``j``.
            P_i (dict): Contribution Predictions.
            n (int): Number of defects on ground truth

        Returns:
            set: C
        """
        c = set()
        for i in range(0, n):
            # i-th set of pixel in D
            di_ = torch.zeros(Di.shape)
            di_[torch.where(Di == i + 1)] = 1
            area_di = torch.sum(di_)
            
            sum_pi = 0
            current_p = P_i[i]
            for j in current_p:
                # j-th set of pixel in P
                pj_ = torch.zeros(Pj.shape)
                pj_[torch.where(Pj == j + 1)] = 1
                
                intersection_di_pj = torch.zeros(di_.shape)
                intersection_di_pj[torch.where(pj_ + di_ == 2)] = 1

                sum_pi += torch.sum(intersection_di_pj) / area_di

            if sum_pi >= self.theta_d:
                c.add(i)
                
        return c
    
    def _compute_F(self, nP_i: dict, n: int) -> list:
        """Computes the false positive predictions from the given set/dict: ``nP_i``.

        Args:
            nP_i (dict): Non-contributing predictions to defect i.
            n (int): Number of defects on ground truth

        Returns:
            list: F
        """
        Fi = [i for i in range(n)]
        
        for cand in Fi[::-1]:
            for nPi_idx in nP_i.keys():
                if not cand in nP_i[nPi_idx] and cand in Fi:
                    Fi.pop(cand)
                    
        return Fi
    
    def _compute_single_rdr_value(self, preds: torch.Tensor, target: torch.Tensor) -> float:
        """Computes the RDR (Region Detection Rate) value for a single pair of prediction and 
        ground truth map.

        Args:
            preds (torch.Tensor): Prediction map
            target (torch.Tensor): Ground truth map

        Returns:
            float: RDR value
        """
        #########################################
        #       Extract objects from maps       #
        #########################################
        # Needs about 10 sec. for each Seq. Labelling
        Di = SequentialLabeling(target)
        Pj = SequentialLabeling(preds)
        
        n: int = int(torch.max(Di))
        m: int = int(torch.max(Pj))
        
        ###########################################
        #       Calculating P(i) and nP(i)        #
        ###########################################
        P_i, nP_i =self._compute_Pi_nPi(Di, Pj, n, m)
        
        ##############################
        #       Calculating Ci       #
        ##############################
        c = self._compute_C(Di, Pj, P_i, n)
        
        ##################################
        #       Calculation of Fi        #
        ##################################
        f = self._compute_F(nP_i=nP_i, n=n)
        
        try:
            rdr_val = len(c)/ (n + self.alpha * len(f))
        except ZeroDivisionError:
            # print(tc.warn + f"Couldn't calculate RDR Value! n={n}, \u03B1={self.alpha}, |f|={len(f)}")
            rdr_val = 0    
            
        self.rdr_sum += rdr_val
        return float(rdr_val)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Computes the RDR (Region Detection Rate) values for all grayscale images in the Batch.
        Returns the RDR Values as float or as Tensor, depending on what the input Tensor is [(H, W) vs. (N, H, W)].
        
        The ``compute`` function will then compute the average.

        Args:
            preds (torch.Tensor): Prediction map(s).
            target (torch.Tensor): Ground truth map(s).

        Raises:
            ValueError: If the shape of the input Tensors doesn't match the expected format.

        Returns:
            float or torch.Tensor: RDR values
        """
        #############################
        #       Input checks        #
        #############################
        assert preds.shape == target.shape
        
        if len(preds.shape) > 3 or len(preds.shape) == 1:
            raise ValueError(tc.err + f"Predictions and target maps have to be of the form: (N, H, W) or (H, W)! But got {preds.shape}")
        
        rdr_values = 42
        if len(preds.shape) == 3:
            # Shape is (N, H, W)
            num, _, _ = preds.shape
            self.num_imgs += num
            
            rdr_values = torch.zeros(num)
            for k in range(0, num):
                rdr_values[k] = self._compute_single_rdr_value(preds[k, ...], target[k, ...])
        else:
            # Shape is (H, W)
            self.num_imgs += 1
            rdr_values = self._compute_single_rdr_value(preds, target)   
        
    def compute(self) -> float:
        """Computes the average from the previously calculated RDR values in the ``update``-step.

        Returns:
            float: Avg. RDR value
        """
        return self.rdr_sum.float() / self.num_imgs.float()


# Helper functions for testing
def drawCircle(img: torch.Tensor, r: int, a: int, b: int, value: float) -> torch.Tensor:
    """Draws and fills a circle at the position ``(a, b)`` with a radius ``r`` on the given ``img``.

    Args:
        img (torch.Tensor): Input image.
        r (int): Radius of circle.
        a (int): Center x position of circle.
        b (int): Center y position of circle.

    Returns:
        torch.Tensor: Image with a filled circle.
    """
    h, w = img.shape
    local_img = img.clone()
    
    x = torch.arange(0, w)
    y = torch.arange(0, h)
    
    # Switch x and y, because indexing of tensor is: arr[y, x] not arr[x, y]
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    
    z = (grid_x - a)**2 + (grid_y - b)**2
    idx = torch.where(z <= r**2)
    
    local_img[idx] = value
    return local_img
    

if __name__ == '__main__':
    IMG_SIZE = 512
    R = 70
    
    #####################################
    #       Generate Test images        #
    #####################################
    gt = torch.zeros((IMG_SIZE, IMG_SIZE))
    gt = drawCircle(gt, r=R, a=150, b=150, value=1.0)
    gt = drawCircle(gt, r=R, a=375, b=375, value=1.0)
    
    case1 = torch.zeros((IMG_SIZE, IMG_SIZE))
    case1 = drawCircle(case1, r=50, a=150, b=150, value=1.0)
    case1 = drawCircle(case1, r=50, a=375, b=375, value=1.0)
    
    case2 = torch.zeros((IMG_SIZE, IMG_SIZE))
    case2 = drawCircle(case2, r=R, a=150, b=150, value=1.0)
    
    case3 = torch.zeros((IMG_SIZE, IMG_SIZE))
    case3 = drawCircle(case3, r=R, a=150, b=150, value=1.0)
    case3 = drawCircle(case3, r=R, a=375, b=375, value=1.0)
    
    idx = torch.where(case3 == 1.0)
    # get size
    random_numbers = torch.rand(idx[0].shape) > 0.75
    case3[idx] = 1 - random_numbers.type(torch.float)

    gt_case4 = torch.zeros((IMG_SIZE, IMG_SIZE))
    gt_case4 = drawCircle(gt_case4, r=R, a=150, b=150, value=1.0)
    gt_case4 = drawCircle(gt_case4, r=R, a=375, b=375, value=1.0)
    gt_case4 = drawCircle(gt_case4, r=R, a=150, b=375, value=1.0)
    
    case4 = torch.zeros((IMG_SIZE, IMG_SIZE))
    case4 = drawCircle(case4, r=R, a=150, b=150, value=1.0)
    case4 = drawCircle(case4, r=50, a=375, b=375, value=1.0)
    case4 = drawCircle(case4, r=R, a=375, b=150, value=1.0)
    
    #############################
    #       Calculate RDR       #
    #############################
    rdr_metric = RDR()
    rdr_case1 = rdr_metric.update(case1, gt)
    print(tc.train + f"RDR(Case1)={rdr_metric.compute()}")
    rdr_case4 = rdr_metric.update(case4, gt_case4)
    print(tc.train + f"RDR(Case4)={rdr_metric.compute()}")
    rdr_value = rdr_metric.compute()
    
    print(tc.train + f"RDR Value: {rdr_value}")
    
    print(tc.success + "finished!")