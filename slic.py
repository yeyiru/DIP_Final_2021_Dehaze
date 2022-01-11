import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.segmentation import slic, mark_boundaries
from skimage.restoration import denoise_bilateral
from skimage.measure import regionprops
from skimage.morphology import opening, closing, disk
from skimage.exposure import equalize_hist
from skimage.color import rgb2gray
from skimage import io

from utils.general import dcp_A, de_haze, get_edge, ToWhite, normalization
from utils.filter import min_filter, guided_filter, gamma_filter
from utils.metrics import metrics

def d_map(density, delta):
    # 當濃度小於某一個閾值時保留原來的濃度，否則以閾值替代，避免過度
    d_map = np.where(density<=delta, density, np.ones(density.shape, dtype=np.float32) * delta)
    return d_map

def get_A_edge(dark_channel, img, threshold=0.07, t_delta=0.95, guided_r=68):
    A = np.zeros((3), dtype=np.float32)
    edges = get_edge((img * 255).astype(np.uint8))
    density = ToWhite(edges)
    density = normalization(density)
    
    # 按照你的方法算出来的density雾越浓的地方越黑，所以用1-，讓越濃的地方越白，更符合直覺
    d = d_map(1 - density, t_delta)
    # 導向濾波平滑化
    t_weight = guided_filter(d, dark_channel, r=guided_r, eps=0.001)
    plt.title('t_weight')
    plt.imshow(t_weight, cmap='gray')
    plt.show()
    
    # 根據density 保留最濃的地方用來估算A
    img[:, :, 0] = np.where(density <= threshold, img[:, :, 0], np.zeros(dark_channel.shape, dtype=np.float32))
    img[:, :, 1] = np.where(density <= threshold, img[:, :, 1], np.zeros(dark_channel.shape, dtype=np.float32))
    img[:, :, 2] = np.where(density <= threshold, img[:, :, 2], np.zeros(dark_channel.shape, dtype=np.float32))
    dark_channel = np.where(density <= threshold, dark_channel, np.zeros(dark_channel.shape, dtype=np.float32))

    plt.title('Clip Image for A')
    plt.imshow(img)
    plt.show()
    
    dark_flat = dark_channel.flatten()
    max_idx = np.argsort(-dark_flat, axis=0)[:int(len(dark_flat)*0.001 + 1)]
    
    img_flat = img.reshape(img.shape[0] * img.shape[1], 3)
    A[0] = img_flat[max_idx, 0].max()
    A[1] = img_flat[max_idx, 1].max()
    A[2] = img_flat[max_idx, 2].max()
    return A, t_weight

# 通過SLIC改進DCP方法
def slic_dcp(img, n_segments, min_k, guided_r):
    # # 按照論文計算局部A
    # def get_A(img):
    #     kernel = disk(10)
    #     # 計算每個pixel RGB的最大值
    #     A_c = img.max(axis=2)

    #     A = np.ones(img.shape, dtype=np.float32)
    #     # 原來想試用SLIC估算局部A
    #     segments = slic(A_c, n_segments=n_segments, compactness=10)
    #     regions = regionprops(segments)
    #     t = np.where(segments == 0)
    #     img_reg = A_c[np.where(segments == 0)[0], np.where(segments == 0)[1]]
    #     A[np.where(segments == 0)[0], np.where(segments == 0)[1], :] = img_reg.max()
    #     for prop in regions:
    #         seg_arr = prop.coords
    #         img_reg = A_c[seg_arr[:, 0], seg_arr[:, 1]]
    #         A[seg_arr[:, 0], seg_arr[:, 1], :] = img_reg.max()
        
    #     # 導向濾波平滑化
        
    #     A[:, :, 0] = guided_filter(A[:, :, 0], A_c, r=38, eps=0.001)
    #     A[:, :, 1] = guided_filter(A[:, :, 1], A_c, r=38, eps=0.001)
    #     A[:, :, 2] = guided_filter(A[:, :, 2], A_c, r=38, eps=0.001)
    #     plt.imshow(A[:, :, 1], cmap='gray')
    #     plt.show()
    #     return A

    def get_t(A, img, w=0.95):
        # Slic方法
        t = np.ones(img.shape, dtype=np.float32) * -1
        if isinstance(w, float):
            w = np.ones(img.shape[:2], dtype=np.float32) * w
        w = np.dstack([w, w, w])
        
        # 分割
        segments = slic(img, n_segments=n_segments, compactness=10, start_label=0)
        regions = regionprops(segments)
        # 不知道為什麼使用regionprops遍歷segments時候會漏掉第一個polygen，所以直接指定第一個polygen賦值
        img_reg = img[np.where(segments == 0)[0], np.where(segments == 0)[1], :]
        w_reg = w[np.where(segments == 0)[0], np.where(segments == 0)[1], :]

        t[np.where(segments == 0)[0], np.where(segments == 0)[1], :] = 1 - (w_reg * img_reg.min(axis=0) / A[np.where(segments == 0)[0], np.where(segments == 0)[1], :])
        
        # 遍歷其他的分割塊，依次計算其每個的t值
        for prop in regions:
            seg_arr = prop.coords
            img_reg = img[seg_arr[:, 0], seg_arr[:, 1], :]
            w_reg = w[seg_arr[:, 0], seg_arr[:, 1], :]
            t[seg_arr[:, 0], seg_arr[:, 1], :] = 1 - (w_reg * img_reg.min(axis=0) / A[seg_arr[:, 0], seg_arr[:, 1], :])
        return t
    
    # 按照論文實現t的微調，但這塊工作一直不太正常感覺
    # def refine_t(img, A, t, alpha=0.95):
    #     t_last = np.zeros(img.shape, dtype=np.float32)
    #     map_0 = img - alpha * A

    #     for c in range(map_0.shape[2]):
    #         t_c = np.minimum(np.sum(map_0[:, :, c] > 0.) / (img.shape[0] * img.shape[1]), 0.28)
    #         t_last[:, :, c] = np.where(abs(img[:, :, c] - A[c]) >= t_c,
    #                                     t[:, :, c],
    #                                     np.minimum(t_c / abs(img[:, :, c] - A[c]), np.ones(t_c.shape, dtype=np.float32)))
    #     return t_last
    
    dark_channel = min_filter(img, min_k)
    # 計算全局大氣光A和t_weight，都要用到density所以寫在一起
    A, t_weight = get_A_edge(dark_channel, img.copy())
    # A = dcp_A(dark_channel, img, mutil_A=True)
    A = np.ones(img.shape, dtype=np.float32) * A
    
    # 計算局部大氣光A
    # A = get_A(img)
    
    # 計算t
    t = get_t(A, img, t_weight)
    
    # 導向濾波平滑t
    t[:, :, 0] = guided_filter(t[:, :, 0], t[:, :, 0], r=guided_r, eps=0.001)
    t[:, :, 1] = guided_filter(t[:, :, 1], t[:, :, 1], r=guided_r, eps=0.001)
    t[:, :, 2] = guided_filter(t[:, :, 2], t[:, :, 2], r=guided_r, eps=0.001)
    
    plt.title('t')
    plt.imshow(t[:, :, 0], cmap='gray')
    plt.show()
    
    # t_last = refine_t(img, A, t)
    
    res = de_haze(img, A, t)
    # res_refine = de_haze(img, A, t_last)
    return res # , res_refine

hazy = io.imread("./Data/HAZY/2.png")[:, :, :3] / 255
gt = io.imread("./Data/GT/2.png")
# gt = io.imread("./Data/HAZY/6.png")

#                 分割塊數，最小值濾波ksize 導向濾波半徑
res = slic_dcp(hazy, 5000, 39, 99)
_ = metrics(4, res, gt, 'SLIC') 
# _ = metrics(4, res_refine, gt, 'SLIC_Refine')

_, ax = plt.subplots(1, 4, figsize=(15, 15))

ax[0].set_title('Original')
ax[0].imshow(hazy)

ax[1].set_title('GT')
ax[1].imshow(gt)

ax[2].set_title('Dehaze_SLIC')
ax[2].imshow(res)

ax[3].set_title('Dehaze_SLIC_Refine')
ax[3].imshow(gamma_filter(res, 0.5))

plt.show()

