import os
import cv2

import numpy as np

from PIL import Image

from skimage import io
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from skimage.color import deltaE_ciede2000, rgb2lab

from prettytable import PrettyTable

RGB2YRGBY = np.array([
    [0.2714, 0.6938, 0.0638], 
    [-0.0971, 0.1458, -0.0250], 
    [-0.0930, -0.2529, 0.4665]
])


def UIQI(RGBimg1, RGBimg2):
    # convert rgb image into gray scale image
    Grayimg1 = np.array(Image.fromarray( RGBimg1 ).convert('L'))
    Grayimg2 = np.array(Image.fromarray( RGBimg2 ).convert('L'))

    # image size
    img_size = Grayimg1.shape[0]*Grayimg1.shape[1]

    # 
    Xbar = Grayimg1.mean()
    Ybar = Grayimg2.mean()
    Xvar = ( (Grayimg1 - Xbar)**2 ).sum()*( 1/(img_size-1) )
    Yvar = ( (Grayimg2 - Ybar)**2 ).sum()*( 1/(img_size-1) )
    XYvar = ( (Grayimg1 - Xbar)*(Grayimg2 - Ybar) ).sum()*( 1/(img_size-1) )
    
    # compute UIQI
    Q = ( 4 * XYvar * Xbar * Ybar )/( (Xvar + Yvar)*(Xbar**2 + Ybar**2) )

    return Q

def CIQI(RGBimg1, RGBimg2):
    RGBimg1 = (RGBimg1 * 255).astype(np.uint8)
    RGBimg2 = (RGBimg2 * 255).astype(np.uint8)
    # get UIQI of RGBimg1 and RGBimg2
    uiqi = UIQI(RGBimg1, RGBimg2)

    # image size
    img_shape = RGBimg1.shape

    # convert rgb image into gray Y, RG, BY
    YRGBY1 = np.matmul(RGBimg1.reshape(-1, 3), RGB2YRGBY.T).reshape(img_shape)
    YRGBY2 = np.matmul(RGBimg2.reshape(-1, 3), RGB2YRGBY.T).reshape(img_shape)

    # compute rRG, rBY
    RGnor1 = YRGBY1[:, :, 1] - YRGBY1[:, :, 1].mean()
    RGnor2 = YRGBY2[:, :, 1] - YRGBY2[:, :, 1].mean()
    BYnor1 = YRGBY1[:, :, 2] - YRGBY1[:, :, 2].mean()
    BYnor2 = YRGBY2[:, :, 2] - YRGBY2[:, :, 2].mean()
    
    rRG = ( (RGnor1*RGnor2).sum() ) / ( (RGnor1**2).sum()*(RGnor2**2).sum() )**0.5
    rBY = ( (BYnor1*BYnor2).sum() ) / ( (BYnor1**2).sum()*(BYnor2**2).sum() )**0.5

    # compute CIQI
    CIQI = 0.5327*uiqi + 0.49774*rRG - 0.25386*rBY

    return CIQI, rRG, rBY

def img_pre(gt, de_hazy):
    # 圖像歸一，並轉為float32
    if de_hazy.shape[-1] > 3:
        de_hazy = de_hazy[:, :, :3]
    if gt.shape[-1] > 3:
        gt = gt[:, :, :3]
    
    if (de_hazy > 1).any():
        de_hazy = de_hazy / 255
    if (gt > 1).any():
        gt = gt / 255

    return gt.astype(np.float32), de_hazy.astype(np.float32)

def metrics(idx, de_hazy, gt, method=''):
    gt, de_hazy = img_pre(de_hazy, gt)
    
    mse = mean_squared_error(gt, de_hazy)
    ssim = structural_similarity(gt, de_hazy, multichannel=True, data_range=1.)
    psnr = peak_signal_noise_ratio(gt, de_hazy, data_range=1.)
    deltaE = deltaE_ciede2000(rgb2lab(gt), rgb2lab(de_hazy)).mean()
    ciqi = CIQI(gt, de_hazy)

    print(f'Image: {idx}.png')
    metrics_table = PrettyTable(['Method', 'MSE↓', 'SSIM↑', 'PSNR↑', '△ E↓', 'CIQI'])
    metrics_table.add_row([method, format(mse, '.5f'), format(ssim, '.5f'), format(psnr, '.5f'), format(deltaE, '.5f'),
                           format(np.mean(ciqi), '.5f')])
    print(metrics_table)
    
    return idx, method, mse, ssim, psnr, deltaE, ciqi

if __name__=='__main__':
    de_hazy = io.imread('F:\\DIP\\Final\\Data\\HAZY\\4.png')
    gt = io.imread('F:\DIP\Final\Data\\4.png')
    metrics(4, de_hazy, gt, method='SLIC')