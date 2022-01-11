import cv2
import numpy as np
from sklearn import preprocessing

def get_edge(img):
    density = np.ones((img.shape[0], img.shape[1], 1)) * -1
    find = False
    for i in range(1010, 0, -1):
        edges = cv2.Canny(img, 1, i)
        if (edges == 0).all():
            continue
        else:
            if not find:
                density[edges == 255] = i
                former = edges.copy()
                find = True
            else:
                this = edges - former
                density[this == 255] = i
                former = edges.copy()
    density = density[:, :, 0]
    return density

def ToWhite(EdgeValue, N=5):
    W = EdgeValue.shape[1]
    H = EdgeValue.shape[0]
    rank = list(np.linspace(1, 1000, 20))
    density = np.zeros((H, W))
    for i in range(int(H / N)):
        for j in range(int(W/N)):
            image = EdgeValue[i*N:i*N+(N-1),j*N:j*N+(N-1)]
            MIN = 0
            index = -1
            for k in range(len(rank)-1):
                if len(image[(image > rank[k]) & (image <= rank[k+1])]) > MIN:
                    MIN = len(image[(image > rank[k]) & (image <= rank[k+1])])
                    index = k + 1
            if len(image[(image < rank[0]) & (image > 0)]) > MIN:
                MIN = len(image[image < rank[0]])
                index = 0
            if len(image[image > rank[len(rank) - 1]]) > MIN:
                MIN = len(image[image > rank[len(rank) - 1]])
                index = 20
            if index == -1:
                index = 0
            density[i*N:i*N+N,j*N:j*N+N] = index*(1/20)
    density = density*256
    return density

# DCP方法估算全局A，分RGB通道估算
def dcp_A(dark_channel, img, mutil_A= True):
    A = np.zeros((3), dtype=np.float32)
    dark_flat = dark_channel.flatten()
    max_idx = np.argsort(-dark_flat, axis=0)[:int(len(dark_flat)*0.001 + 1)]
    
    img_flat = img.reshape(img.shape[0] * img.shape[1], 3)
    A[0] = img_flat[max_idx, 0].max()
    A[1] = img_flat[max_idx, 1].max()
    A[2] = img_flat[max_idx, 2].max()
    if not mutil_A:
        A = np.array([A.max(), A.max(), A.max()], dtype=np.float32)
    return A

# 归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
 
# 标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def de_haze(raw_img, A, t, t0=0.1):
    t0 = np.ones(t.shape, dtype=np.float32) * t0
    out = (raw_img - A) / np.maximum(t, t0) + A
    # 打印t的最大最小值及out的最大最小值
    print(f't_max={t.max()}')
    print(f't_min={t.min()}')
    print(out.max())
    print(out.min())
    # out = np.clip(out, 0., 1.)
    out = normalization(out)
    
    return out