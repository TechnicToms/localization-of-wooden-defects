import numpy as np
from multiprocessing import Process, Manager
import cv2
from helpers.terminalColor import terminalColor as tc

class salience_metric:
    """Calculates the salience metric for the given parameters and image
    """
    def __init__(self, K: int=12, scales: list=[1, 0.8, 0.66, 0.4]) -> None:
        """Init of salience metric class

        Args:
            K (int, optional): Neighborhood size. Defaults to 12.
            scales (list, optional): list of scales (from 0 to 1). Defaults to [1, 0.8, 0.66, 0.4].
        """
        self.K = K
        self.scales = scales

    def __repr__(self) -> str:
        return tc.info + f"Salience Metric(K: {self.K}, scales: {self.scales})"

    def extractFeatures(self, rgb: np.ndarray, step: int) -> np.ndarray:
        """returns feature Matrix with dimension (x, y, 147)

        Args:
            rgb (np.ndarray): Input Image as type RGB
            step (int): Steps to go [0-7]. Defines how much overlap between the features are used

        Returns:
            np.ndarray: features
        """
        # Convert to CIE L*a*b
        Lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)

        # Get Size
        w, h, c = Lab.shape

        # Kernel Size and calc output size
        k = 7
        ox = int((w - k) / step) + 1
        oy = int((h - k) / step) + 1

        # Pre-Allocate Memory for Feature Vector
        feature = np.zeros((ox, oy, 147))

        # Split up into 7x7 Areas
        for y in range(0, h, step):
            for x in range(0, w, step):
                if(x+7 <= w and y+7 <= h):
                    L = Lab[x:x+7, y:y+7, 0]
                    a = Lab[x:x+7, y:y+7, 1]
                    b = Lab[x:x+7, y:y+7, 2]

                    feature[int(x/step), int(y/step), 0:49] = L.reshape(7*7)
                    feature[int(x/step), int(y/step), 49:98] = a.reshape(7*7)
                    feature[int(x/step), int(y/step), 98:147] = b.reshape(7*7)

        return feature

    def d_color(self, features: np.ndarray) -> np.ndarray:
        """Calculates the quadratic euclidean distance between the feature Vectors

        Args:
            features (np.ndarray): feature tensor

        Returns:
            np.ndarray: d_color[x, y, i, j]
        """
        # Get Shape
        wx, hy, _ = features.shape

        # Pre- Allocate Memory to save in d_color
        d = np.zeros((wx, hy, self.K, self.K))

        # Predefine the Radius of the Neighborhood
        radius_k = int(self.K/2)

        # Because of open cv: divide by 255.0
        features = features / 255.0
        # Extend array, so neighborhood can be calculated for the edges too
        wf, hf, cf = features.shape
        new_features = np.ones((wf + self.K, hf + self.K, cf)) * \
            np.mean(features, axis=(0, 1))
        new_features[radius_k:-radius_k, radius_k:-radius_k, :] = features

        # Calculate quadratic euclidean distance
        for i in range(0, wx * hy):
            # Calculate the position in the 2D Grid from the 1D iteration vector
            x = i % wx + radius_k
            y = i // wx + radius_k

            # Pick out the center vector
            mainVec = new_features[x, y, :]
            # Calculate the difference of color
            inner_vec = mainVec - \
                new_features[x - radius_k:x + radius_k,
                            y - radius_k:y + radius_k, :]
            d[x - radius_k, y - radius_k, :, :] = np.sum(inner_vec**2, axis=-1)

        # Fix issue that the edges of the first two edges of the images get more highlighted
        d[0:radius_k, :, :, :] = d[0:radius_k, :, :, :] * 0.5
        d[radius_k:, 0:radius_k, :, :] = d[radius_k:, 0:radius_k, :, :] * 0.5

        return d

    def d_distance(self, features: np.ndarray) -> np.ndarray:
        """Calculates the Distance for each point to neighbor Points

        Args:
            features (np.ndarray): features

        Returns:
            np.ndarray: distance tensor
        """    
        # Get Shape
        wx, hy, _ = features.shape

        # Pre- Allocate Memory to save in d_color
        d = np.zeros((wx, hy, self.K, self.K))

        # Precalculate the distance pattern
        pattern = np.zeros((self.K, self.K))
        for i in range(-int(self.K/2), int(self.K/2)):
            for j in range(-int(self.K/2), int(self.K/2)):
                pattern[i, j] = np.sum((np.array((0, 0)) - np.array((i, j)))**2)

        # Calculate quadratic euclidean distance
        for y in range(0, hy):
            for x in range(0, wx):
                d[x, y, :, :] = pattern

        return d

    def calculate_defectmap(self, rgb: np.ndarray) -> np.ndarray:
        """Calculates the defect map for a given image

        Args:
            rgb (np.ndarray): rgb image

        Returns:
            np.ndarray: defectmap
        """
        # For algorithm
        w, h, c = rgb.shape

        c = 3
        maps = []
        feature_maps = []
        for sc in self.scales:
            # Scale image
            scaled = cv2.resize(rgb, (int(sc * w), int(sc * h)))
            # features needed: ~ 0.285 sec for 300px by 300px
            feature_maps.append(self.extractFeatures(scaled, step=3))
        
        for feature in feature_maps:
            # d color needed: 2.6 sec
            dc = self.d_color(feature)
            # d distance needed: 0.062 sec
            d_dis = self.d_distance(feature)

            d = dc / (1 + c * d_dis)

            defectmap = 1 - 2.718**(-1/self.K*np.sum(d, axis=(2, 3)))
            maps.append(defectmap)

        # Fusion
        length = len(self.scales)
        w, h = maps[0].shape

        defectmap_ = np.zeros((h, w))

        for i in range(0, length):
            scaled = cv2.resize(maps[i], (w, h))
            defectmap_ = defectmap_ + scaled
        
        return defectmap_

    def calculate_defectmap_parallel(self, rgb: np.ndarray) -> np.ndarray:
        """Calculates the defect map for a given image but uses multiprocessing

        Args:
            rgb (np.ndarray): rgb image

        Returns:
            np.ndarray: defectmap
        """
        # For algorithm
        w, h, c = rgb.shape

        maps = []
        feature_maps = []
        for sc in self.scales:
            # Scale image
            scaled = cv2.resize(rgb, (int(sc * w), int(sc * h)))
            # features needed: ~ 0.285 sec for 300px by 300px
            feature_maps.append(self.extractFeatures(scaled, step=3))
        
        # Create a process manager
        manager = Manager()
        return_dict1 = manager.dict()
        # Store running processes in a list
        processes = []
        task_id = 0
        
        for feature in feature_maps:
            processes.append(Process(target=self.__generate_single_scale_defectmap, 
                                     args=[feature, task_id, return_dict1]))
            task_id += 1
            
        # Start all processes
        for t in processes:
            t.start()

        # Collect all Processes
        for t in processes:
            t.join()
        
        # Fusion
        defectmap_ = np.zeros((w, h))

        for i in range(0, task_id):
            scaled = cv2.resize(return_dict1[i], (h, w))
            defectmap_ = defectmap_ + scaled
            
        return defectmap_

    def __generate_single_scale_defectmap(self, feature: np.ndarray, task_id: int, return_dict) -> np.ndarray:
        """_summary_

        Args:
            feature (np.ndarray): feature tensor
            task_id (int): task id for the process
            return_dict (dict): return dictionary for process

        Returns:
            np.ndarray: defectmap in return dictionary
        """
        # d color needed: 2.6 sec
        dc = self.d_color(feature)
        # d distance needed: 0.062 sec
        d_dis = self.d_distance(feature)

        d = dc / (1 + 3 * d_dis)

        return_dict[task_id] = 1 - 2.718**(-1/self.K*np.sum(d, axis=(2, 3)))
    
