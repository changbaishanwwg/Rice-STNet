import numpy as np
import cv2

def replacezeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])  # 找到数组中最小的非零值
    data = np.where(data == 0, min_nonzero, data)  # 将数组中的零值替换为最小的非零值
    return data  # 返回替换后的数组

def get_ksize(sigma):
    return int(((sigma - 0.8)/0.15) + 2.0)

def get_gaussian_blur(img, ksize=0, sigma=5):
    # if ksize == 0, then compute ksize from sigma
    if ksize == 0:
        ksize = get_ksize(sigma)
    sep_k = cv2.getGaussianKernel(ksize, sigma)

    return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))

def ssr(img, sigma):
    src_img = replacezeroes(img)
    return np.log10(src_img) - np.log10(get_gaussian_blur(src_img, ksize=0, sigma=sigma) + 1.0)

def msr(img, sigma_scales):
    msr = np.zeros(img.shape)
    # for each sigma scale compute SSR
    for sigma in sigma_scales:
        msr += ssr(img, sigma)
    msr = msr / len(sigma_scales)

    return msr

def color_balance(img, low_per, high_per):
    """Contrast stretch img by histogram equalization with black and white cap."""
    tot_pix = img.shape[1] * img.shape[0]
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100

    ch_list = [img] if len(img.shape) == 2 else cv2.split(img)
    cs_img = []
    for ch in ch_list:
        cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)))
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if li == hi:
            cs_img.append(ch)
            continue
        lut = np.array([0 if i < li else (255 if i > hi else round((i - li) / (hi - li) * 255))
                        for i in range(256)], dtype='uint8')
        cs_img.append(cv2.LUT(ch, lut))
    return cv2.merge(cs_img) if len(cs_img) > 1 else np.squeeze(cs_img)


def msrcp(img, sigma_scales=[15, 80, 250], low_per=1, high_per=1):
    """
    Multi-Scale Retinex with Color Preservation (MSRCP).
    Parameters:
        - img: Input image
        - sigma_scales: List of sigma values for Gaussian blur
        - low_per, high_per: Percentiles for color balance
    """
    # Compute intensity (Int)
    int_img = np.sum(img, axis=2) / img.shape[2]
    int_img = replacezeroes(int_img)  # Prevent division by zero

    # Multi-scale retinex on intensity
    msr_int = msr(int_img, sigma_scales)
    msr_int = (msr_int - np.min(msr_int)) / (np.max(msr_int) - np.min(msr_int)) * 255.0
    msr_cb = color_balance(msr_int.astype(np.uint8), low_per, high_per)

    # Compute scaling factor (A)
    B = 256.0 / (np.max(img, axis=2) + 15.0)
    A = np.minimum(B, msr_cb / int_img)

    # Compute MSRCP
    msrcp_img = np.clip(np.expand_dims(A, 2) * img, 0.0, 255.0)
    return msrcp_img.astype(np.uint8)
