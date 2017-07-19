import cv2
import numpy as np
from matplotlib import pyplot as plt

class CNN(object):
    def __init__(self):
        self.inputheight = 32
        self.inputwidth = 32
        self.inputdepth = 3
        self.filterDefs = [(3,3,3), (2,2,3)]
        self.poolDefs = [(2,2), (3,3)]
        if(len(self.filterDefs) != len(self.poolDefs)):
           print "Critical Error: Filter and Pool sizes do not match!"
        self.finalPoolSize = self.getFinalPoolSize()
        self.hiddenLayerSize = 100
        self.outputLayerSize = 6

        self.filters = []
        for i, spec in enumerate(self.filterDefs):
            channelMult = self.inputdepth
            if i != 0:
                channelMult = self.filterDefs[i-1][2]
            rows = spec[0] * spec[1] * channelMult

            self.filters.append(np.random.randn(rows, spec[2]))
           
        self.W1 = np.random.randn(self.finalPoolSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)


    def forwardImages(self, images):
        FCInputs = np.zeros([len(images), self.finalPoolSize])
        for i in range(0, len(images)):
            FCInputs[i] = self.convolveImage(images[i]).flatten()

        return self.forwardFC(FCInputs)
        
    def forwardFC(self, inputs):
        z2 = np.dot(inputs, self.W1)
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W2)
        yHat = self.sigmoid(z3)
        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z)/((a+np.exp(-z))**2)

    def costFunction(self, images, y):
        yHat = forwardImages(images)
        J = 0.5 * sum((y - self.yHat)**2)
        
    
    def convolveImage(self, image):
        img = image
        for i in range(0, len(self.filterDefs)):
            img = self.pool(self.applyFilters(img, self.filters[i], self.filterDefs[i][0], self.filterDefs[i][1]), self.poolDefs[i][0], self.poolDefs[i][1])
        return img
            
    def getFinalPoolSize(self):
        height = self.inputheight
        width = self.inputwidth
        depth = self.inputdepth

        for i in range(0, len(self.filterDefs)):
           height += self.filterDefs[i][0] - 1
           width += self.filterDefs[i][1] - 1
           depth = self.filterDefs[i][2]
           height /= self.poolDefs[i][0]
           width /= self.poolDefs[i][1]
        return height*width*depth

        
    def padImage(self, img, heightmargins, widthmargins, r, g, b):
        return cv2.copyMakeBorder(img, heightmargins, heightmargins, widthmargins, widthmargins, cv2.BORDER_CONSTANT, value=[r,g,b])

    def unpadImage(self, img, depth):
        return img[depth:-depth, depth:-depth]

    def plotChannelsMonochrome(self, channelmat, height, width, row=1, desc='Channel'):
        for x in range(0, channelmat.shape[2]):
            filtered = channelmat[:,:,x]
            filtered = filtered / np.amax(filtered)
            
            f2 = filtered.reshape(height, width).copy()
            f3 = np.dstack((f2, f2, f2))
            print f3.shape
            plt.subplot(row,channelmat.shape[2],x+1),plt.imshow(f3,'gray'),plt.title(desc + ' ' + str(x))
        
    def chunk(self, img, filterheight, filterwidth):
        # The chunk matrix will have each row being the "flattened" combination of each chunk, scanned left-to-right top-to-bottom
        # Size will be number of chunks by filterwidth*filterheight*number of channels
        
        m = max(np.amax(img), 1)
        padded = self.padImage(img, filterheight - 1, filterwidth - 1, 0,0,0)
        
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

    def pool(self, img, vscale, hscale):
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

    def applyFilters(self, img, filtermat, filterwidth, filterheight):
        chunkmat = self.chunk(img, filterheight, filterwidth)
        filteredmat = chunkmat.dot(filtermat)
        
        height = img.shape[0] + filterheight - 1
        width = img.shape[1] + filterwidth - 1
        
        return (filteredmat / np.amax(filteredmat)).reshape(height,width,filteredmat.shape[1])


testNet = CNN()

#img = cv2.imread("arrow.png")
    
#newFilter = np.zeros([27,2])
#newFilter[12,1] = 0
#newFilter[12,0] = 1
#newFilter[12,1] = 1
#newFilter[12,2] = 1

#tmp = applyFilters(img, newFilter, 3, 3)
#pooled = pool(tmp, 2, 2)
#stageTwo = np.zeros([8, 1])
#stageTwo[0,0] = 1
#stageTwo[1,0] = 1
#stageTwo[2,0] = 1
#stageTwo[3,0] = 1
#stageTwo[4,0] = 1
#stageTwo[5,0] = 1
#stageTwo[6,0] = 1
#stageTwo[7,0] = 1
#stageTwo[3,0] = -1

#secondPass = applyFilters(pooled, stageTwo, 2, 2)
#pool2 = pool(secondPass, 3, 3)


#plotChannelsMonochrome(pool2, pool2.shape[0], pool2.shape[1])
#plt.show()
