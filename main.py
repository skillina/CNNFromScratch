import cv2
import numpy as np
from matplotlib import pyplot as plt

def padImage(img, heightmargins, widthmargins, r, g, b):
    return cv2.copyMakeBorder(img, heightmargins, heightmargins, widthmargins, widthmargins, cv2.BORDER_CONSTANT, value=[r,g,b])

def unpadImage(img, depth):
    return img[depth:-depth, depth:-depth]

def plotChannelsMonochrome(channelmat, height, width, row=1, desc='Channel'):
    for x in range(0, channelmat.shape[2]):
        filtered = channelmat[:,:,x]
        filtered = filtered / np.amax(filtered)
        
        f2 = filtered.reshape(height, width).copy()
        f3 = np.dstack((f2, f2, f2))
        print f3.shape
        plt.subplot(row,channelmat.shape[2],x+1),plt.imshow(f3,'gray'),plt.title(desc + ' ' + str(x))
        
def chunk(img, filterheight, filterwidth):
    # The chunk matrix will have each row being the "flattened" combination of each chunk, scanned left-to-right top-to-bottom
    # Size will be number of chunks by filterwidth*filterheight*number of channels

    m = max(np.amax(img), 1)
    padded = padImage(img, filterheight - 1, filterwidth - 1, 0,0,0)

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

def pool(img, vscale, hscale):
    pooledmat = np.zeros([img.shape[0]/vscale, img.shape[1]/hscale, img.shape[2]])
    for i in range(0, img.shape[0]/vscale):
        for j in range(0, img.shape[1]/hscale):
            maxes = np.zeros([img.shape[2]])

            for x in range(0, hscale):
                for y in range(0, vscale):
                    for z in range(0, img.shape[2]):
                        maxes[z] = max(maxes[z], img[y + i*vscale, x + j * hscale, z])
            pooledmat[i,j] = maxes
    return pooledmat

def applyFilters(img, filtermat, filterwidth, filterheight):
    chunkmat = chunk(img, filterheight, filterwidth)
    filteredmat = chunkmat.dot(filtermat)
    
    height = img.shape[0] + filterheight - 1
    width = img.shape[1] + filterwidth - 1
        
    return (filteredmat / np.amax(filteredmat)).reshape(height,width,filteredmat.shape[1])

img = cv2.imread("arrow.png")

newFilter = np.zeros([27,2])
#newFilter[12,1] = 0
newFilter[12,0] = 1
newFilter[12,1] = 1
#newFilter[12,2] = 1

tmp = applyFilters(img, newFilter, 3, 3)

plt.subplot(1,1,1),plt.imshow(img,'gray'),plt.title('Original')
plotChannelsMonochrome(tmp, tmp.shape[0], tmp.shape[1], row=2)
pooled = pool(tmp, 2, 2)
plotChannelsMonochrome(pooled, pooled.shape[0], pooled.shape[1], row=3)

stageTwo = np.zeros([8, 1])
stageTwo[0,0] = 1
stageTwo[1,0] = 1
stageTwo[2,0] = 1
stageTwo[3,0] = 1
stageTwo[4,0] = 1
stageTwo[5,0] = 1
stageTwo[6,0] = 1
stageTwo[7,0] = 1
#stageTwo[3,0] = -1

secondPass = applyFilters(pooled, stageTwo, 2, 2)
pool2 = pool(secondPass, 3, 3)
plotChannelsMonochrome(pool2, pool2.shape[0], pool2.shape[1])
plt.show()
