import cv2
import numpy as np
from matplotlib import pyplot as plt

def padImage(img, heightmargins, widthmargins, r, g, b):
    return cv2.copyMakeBorder(img, heightmargins, heightmargins, widthmargins, widthmargins, cv2.BORDER_CONSTANT, value=[r,g,b])

def unpadImage(img, depth):
    return img[depth:-depth, depth:-depth]

def chunk(img, filterheight, filterwidth):
    # The chunk matrix will have each row being the "flattened" combination of each chunk, scanned left-to-right top-to-bottom
    # Size will be number of chunks by filterwidth*filterheight*number of channels

    padded = padImage(img, filterheight - 1, filterwidth - 1, 255, 255, 255)

    rowSize = filterwidth*filterheight*padded.shape[2]
    numchunks = (padded.shape[0]-filterheight + 1) * (padded.shape[1]-filterwidth + 1)

    chunkmat = np.empty([numchunks, rowSize])
    
    for i in range(0, padded.shape[0] - filterheight+1):
        for j in range(0, padded.shape[1] - filterwidth+1):
            rowcontainer = np.empty([filterheight, filterwidth, padded.shape[2]])
            for r in range(0, filterheight):
                for c in range(0, filterwidth):
                    rowcontainer[r][c] = padded[r+i][c+j]
            chunkmat[j + (padded.shape[1]-filterwidth + 1) * i] = rowcontainer.flatten()
            
    return chunkmat
        
    

def applyFilters(img, filtermat, filterwidth, filterheight):
    chunkmat = chunk(img, filterheight, filterwidth)
    filteredmat = chunkmat.dot(filtermat)

    print img.shape
    height = img.shape[0] + filterheight - 1
    width = img.shape[1] + filterheight - 1
    
    for x in range(0, filteredmat.shape[1]):
        filtered = filteredmat[:,x]
        filtered = filtered / max(filtered)
        
        f2 = filtered.reshape(height, width).copy()
        f3 = np.dstack((f2,f2,f2))
        print f3.shape
        plt.subplot(1,filteredmat.shape[1],x+1),plt.imshow(f3,'gray'),plt.title('filter'+str(x))
        
    plt.show()
    return filteredmat
        
image = cv2.imread("square.png")

newFilter = np.zeros([27,2])
newFilter[0,0] = 1
newFilter[3,0] = -1

newFilter[3, 1] = -1
newFilter[6, 1] = 1
applyFilters(image, newFilter, 3, 3)
