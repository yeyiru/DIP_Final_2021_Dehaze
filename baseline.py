import numpy as np
import image_dehazer
import matplotlib.pyplot as plt
import pandas as pd

from skimage import io

from utils.general import dcp_A, de_haze
from utils.filter import min_filter, guided_filter
from utils.metrics import metrics

# He DCP方法
def dcp_he(img, min_k, guided_r):
    def get_t(dark_channel, A, img, w=0.95):
        t = np.zeros(img.shape, dtype=np.float32)
        # A = np.mean(img[:, :, 1], axis=0).max() + np.mean(img[:, :, 0].min() + img[:, :, 2].min())
        t[:, :, 0] = 1 - w * (dark_channel / A[0])
        t[:, :, 1] = 1 - w * (dark_channel / A[1])
        t[:, :, 2] = 1 - w * (dark_channel / A[2])
        return t

    dark_channel = min_filter(img, min_k)
    A = dcp_A(dark_channel, img)
    t = get_t(dark_channel, A, img)
    t[:, :, 0] = guided_filter(t[:, :, 0], t[:, :, 0], guided_r, eps=0.001)
    t[:, :, 1] = guided_filter(t[:, :, 1], t[:, :, 1], guided_r, eps=0.001)
    t[:, :, 2] = guided_filter(t[:, :, 2], t[:, :, 2], guided_r, eps=0.001)
    plt.imshow(t[:, :, 0], cmap='gray')
    plt.show()
    res = de_haze(img, A, t)
    return res

# Meng 方法
def bccr_meng(img):
    img = (img * 255).astype(np.uint8)
    img = img[:,:,::-1]
    res = image_dehazer.remove_haze(img)
    return res[...,::-1]

if __name__=='__main__':
    hazy = io.imread("./Data/HAZY/2.png")[:, :, :3] / 255
    gt = io.imread("./Data//GT//2.png")
    ms = pd.DataFrame([], columns = ['k', 'mse', 'ssim', 'psnr', 'deltaE', 'ciqi'])
    res_he = dcp_he(hazy, 38, 68)
    m = metrics(4, res_he, gt, method='HE')
    
    res_meng = bccr_meng(hazy)
    m = metrics(4, res_meng, gt, method='MENG')
    _, ax = plt.subplots(1, 4, figsize=(15, 15))
    ax[0].set_title('Original')
    ax[0].imshow(hazy)

    ax[1].set_title('GT')
    ax[1].imshow(gt)

    ax[2].set_title('Dehaze_He')
    ax[2].imshow(res_he)

    ax[3].set_title('Dehaze_Meng')
    ax[3].imshow(res_meng)
    plt.show()