import torch
from .terminalColor import terminalColor as tc


def SequentialLabeling(img: torch.Tensor) -> torch.Tensor:
    """ Performs the Sequential Labeling Algorithm on a binary image I, 
    where regions of value 1 will be detected.

    Implementation follows Algorithm described in: 
    
    Burger, W., Burge, M.J. (2009). Regions in Binary Images. In: Principles of Digital Image Processing. Undergraduate Topics in Computer Science. Springer, London. https://doi.org/10.1007/978-1-84800-195-4_2

    Args:
        I (torch.Tensor): Binary input image

    Returns:
        torch.Tensor: Image with regions
    """
    # Clone the image
    I = img.clone().type(torch.int16)
    m, C = AssignInitialLabels(I)
    R = ResolveLabelCollisions(m, C)
    RelabelImage(I, R, m)
    return I
    
def AssignInitialLabels(I: torch.Tensor):
    """Performs a preliminary labeling on image I (which is modified).
    Returns the number of assigned labels (m) and the set of detected label collisions (C).

    Args:
        I (torch.Tensor): Binary input image
    """
    m: int = 2
    C: list = []
    
    h, w = I.shape
    
    # Select all indexes, where the image is 1 (Regions)
    # To iterate in the "right way" transpose the image
    idx = torch.where(I.T == 1)
    
    for i in range(0, len(idx[0])):
        # Load next u and v (this skips all the 0's in between)
        v = idx[0][i]  
        u = idx[1][i]
        
        # Extract Neighbors
        n1 = n2 = n3 = n4 = 0
        
        # Check if indices aren't out of range!
        if u-1 >= 0:
            n1 = I[u-1, v]
            
        if u-1 >= 0 and v-1 >= 0:
            n2 = I[u-1, v-1]
        
        if v-1 >= 0:
            n3 = I[u,   v-1]
            if u+1 < h:
                n4 = I[u+1, v-1]

        # Refactor to a single Tensor
        nk = torch.Tensor([n1, n2, n3, n4])
        
        # Check which indexes are non-zero -> possible candidates for a existing region
        idx_ns = torch.nonzero(nk)
                
        # All neighbors are zero
        if len(idx_ns) == 0:
            I[u, v] = m
            m += 1
        
        # One neighbor has a label value nk > 1
        elif len(idx_ns) == 1:
            I[u, v] = nk[idx_ns]
        
        # Several neighbors of (u, v) have label values nj > 1
        else:
            k = nk[idx_ns[0]]
            I[u, v] = k
            
            # Loop over all label values
            for idx_ni in idx_ns:
                # Check if the new label isn't already in the list
                if int(k) != int(nk[idx_ni]):
                    ci = (int(k), int(nk[idx_ni]))
                
                    # Append only, if the current example isn't already in the list
                    if not ci in C:
                        C.append(ci) 

    return m, C

def ResolveLabelCollisions(m: int, C: list):
    """Resolves the label collisions contained in the set C.
    Returns R, a vector of sets that represents a partitioning
    of the complete label set into equivalent labels

    Args:
        m (int): Number of regions
        C (list): List of collisions 
    """
    L = torch.arange(2, m)
    R = [[int(i)] for i in L]
    
    if len(R) < 2:
        return 
    
    for ci in C:
        a, b = ci

        # Check if Ra and Rb are not the same        
        if not R[a - 2] == R[b - 2]:
            R[a - 2] = R[a - 2] + R[b - 2]
            R[b - 2] = []
        
    return R

def RelabelImage(I: torch.Tensor, R: list, max_m: int):
    """Relabels the image I using the label partitioning in R.
    The image I is modified.

    Args:
        I (torch.Tensor): Processed image I
        R (list): Regions, that are connected
        max_m (int): Maximum number of regions that were defined by previous process
    """
    m_new: int = max_m + 1
    
    if R:
        for ri in R:
            if not ri: 
                continue
            
            for val in ri:
                I[I == val] = m_new
                
            m_new = m_new + 1
            
    I[I>0] = I[I>0] - max_m


if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    
    img = torch.zeros((512, 512))
    # Two Boxes
    img[100:200, 200:300] = 1
    img[300:400, 100:200] = 1
    
    # Create U shape
    img[0:10, 5] = 1
    img[10, 5:11] = 1
    img[0:10, 10] = 1
    
    # Create regions    
    out = SequentialLabeling(img)
    
    plt.imshow(out)
    plt.show()

    print(tc.success + "finished!")