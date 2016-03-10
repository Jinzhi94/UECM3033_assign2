import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sp
import copy as cp
from PIL import Image

#defining function for compressing and decomposing
def compress(U,sigma,V,n):  #a function to compress the image into lower resolution
    sigma_low=cp.copy(sigma)
    for i in range(n,len(sigma)):   #To keep the first n nonzero element in the matrix sigma
        sigma_low[i][i] = 0 
    
    A_low = np.asmatrix(U) * np.asmatrix(sigma_low) * np.asmatrix(V) #combining the three component
    A_low = np.uint8(np.array(A_low))
    return A_low


def decompose(A):   #a function to perform svd decomposition and retun U, sigma, and V
    U,S,V = np.linalg.svd(A)
    sigma = sp.linalg.diagsvd(S,len(A[:,0]),len(A[0,:]))
    return U,sigma,V



img=mpimg.imread('img.jpg')
[r,g,b] = [img[:,:,i] for i in range(3)]


fig = plt.figure(1)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(img)
ax2.imshow(r, cmap = 'Reds')
ax3.imshow(g, cmap = 'Greens')
ax4.imshow(b, cmap = 'Blues')
plt.show()
fig.savefig('OriImagePlot.jpg')
    
Ured,Sred,Vred = decompose(r)       #decomposing red color
Ugreen,Sgreen,Vgreen = decompose(g) #decomposing green color
Ublue,Sblue,Vblue = decompose(b)    #decomposing blue color


#Counting the number of nonzero element in matrix sigma for each color
counterR = 0
counterG = 0
counterB = 0

for i in range(len(r[:,0])):
    for j in range(len(r[0,:])):
        if Sred[i][j] != 0:
            counterR = counterR + 1
            
for i in range(len(g[:,0])):
    for j in range(len(g[0,:])):
        if Sgreen[i][j] != 0:
            counterG = counterG + 1
            
for i in range(len(b[:,0])):
    for j in range(len(b[0,:])):
        if Sblue[i][j] != 0:
            counterB = counterB + 1
            
print(counterR)
print(counterG)
print(counterB)


#compressing the image into lower resolution, n = 30
r_30 = compress(Ured,Sred,Vred,30)
g_30 = compress(Ugreen,Sgreen,Vgreen,30)
b_30 = compress(Ublue,Sblue,Vblue,30)
img_30 = np.dstack((r_30,g_30,b_30))

fig = plt.figure(2)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(img_30)
ax2.imshow(r_30, cmap = 'Reds')
ax3.imshow(g_30, cmap = 'Greens')
ax4.imshow(b_30, cmap = 'Blues')
plt.show()
fig.savefig('ImageLowPlot.jpg')
img_low=Image.fromarray(img_30,'RGB')
img_low.save('ImageLow.jpg')


#compressing the image into lower resolution, n = 200
r_200 = compress(Ured,Sred,Vred, 200)
g_200 = compress(Ugreen,Sgreen,Vgreen, 200)
b_200 = compress(Ublue,Sblue,Vblue, 200)
img_200 = np.dstack((r_200,g_200,b_200))
            

fig = plt.figure(3)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(img_200)
ax2.imshow(r_200, cmap = 'Reds')
ax3.imshow(g_200, cmap = 'Greens')
ax4.imshow(b_200, cmap = 'Blues')
plt.show()
fig.savefig('ImageBetterPlot.jpg')
img_better=Image.fromarray(img_200,'RGB')
img_better.save('ImageBetter.jpg')





