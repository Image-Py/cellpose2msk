import numpy as np
import scipy.ndimage as ndimg

def flow2msk(flow, prob, grad=1.0, area=150, volume=500):
    l = np.linalg.norm(flow, axis=-1)
    flow /= l[:,:,None]; flow[l<grad] = 0
    flow[[0,-1],:,0], flow[:,[0,-1],1] = 0, 0
    sn = np.sign(flow); sn *= 0.5; flow += sn;
    dn = flow.astype(np.int32).reshape(-1,2)
    strides = np.cumprod(flow.shape[::-1])//2
    dn = (dn * strides[-2::-1]).sum(axis=-1)
    rst = np.arange(flow.size//2) + dn
    for i in range(10): rst = rst[rst]
    hist = np.bincount(rst, minlength=len(rst))
    hist.shape = rst.shape = flow.shape[:2]
    lab, n = ndimg.label(hist, np.ones((3,3)))
    areas = np.bincount(lab.ravel())
    weight = ndimg.sum(hist, lab, np.arange(n+1))
    msk = (areas<area) & (weight>volume)
    lut = np.zeros(n+1, np.int32)
    lut[msk] = np.arange(1, msk.sum()+1)
    mask = lut[lab].ravel()[rst]
    return hist, lut[lab], mask

if __name__ == '__main__':
    from skimage.io import imread, imsave
    import matplotlib.pyplot as plt
    from time import time
    
    flow = imread('1.tif').transpose(1,2,0)#[150:150+256,250:506]
    water, core, msk = flow2msk(flow, None, 1.0, 200, 400)
    ax1, ax2, ax3 = plt.subplot(131), plt.subplot(132), plt.subplot(133)
    ax1.imshow(np.log(water+1))
    ax2.imshow(core)
    ax3.imshow(msk)
    plt.show()
