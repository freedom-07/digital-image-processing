import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image_and_histogram(image, title):
    plt.figure(figsize=(12, 5))

    # 判断图像是否为单通道
    is_gray = len(image.shape) == 2

    # 显示图像
    plt.subplot(121)
    if is_gray:
        plt.imshow(image, cmap='gray')  # 单通道图像使用灰度颜色映射
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

    # 显示直方图
    plt.subplot(122)
    if is_gray:
        # 单通道图像只有一个直方图
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
    else:
        # 多通道图像有三个直方图
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
    plt.title('Histogram')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

def histogram_equalization(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def mean_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def unsharp_masking(image, sigma=0.5, strength=10.0, threshold=0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    high_pass = cv2.subtract(image, blurred)
    high_pass = cv2.threshold(high_pass, threshold, 255, cv2.THRESH_TOZERO)[1]
    sharpened = cv2.addWeighted(image, 1, high_pass, strength, 0)
    return sharpened

def resize_and_show(image, scale=0.5):
    # 缩放图像以适应屏幕
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("Resized Image", resized)
    return resized

def scale_rect(rect, scale):
    # 对矩形框的坐标和尺寸进行缩放
    x, y, w, h = rect
    return (int(x / scale), int(y / scale), int(w / scale), int(h / scale))

# 读取图像
image = cv2.imread('D:\\pythonproject\\work1\\image7.png')
def apply_foreground_highlight_with_blur(image, mask, blur_strength=51):
    # 将mask转换为3通道用于与原图像混合
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 创建背景mask（背景部分为1，前景部分为0）
    background_mask = 1 - mask_3ch

    # 对背景应用高斯模糊
    blurred_background = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

    # 使用背景mask将模糊的背景与原图像混合
    blurred_background = blurred_background * background_mask

    # 使用前景mask将原图像的前景部分提取出来
    foreground = image * mask_3ch

    # 将模糊的背景与清晰的前景相加得到最终图像
    result = blurred_background + foreground

    return result.astype(np.uint8)


def foreground_background_segmentation_grabcut(image, rect=None):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    if rect is not None:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    else:
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask


# 显示缩放后的图像并让用户选择前景矩形
resized_image = resize_and_show(image)
rect = cv2.selectROI("Select ROI", resized_image)
cv2.destroyAllWindows()

# 将矩形框的坐标和尺寸缩放回原始图像大小
scale_factor = 0.5  # 这是之前缩放图像的因子
scaled_rect = scale_rect(rect, scale_factor)

# 使用用户选择的矩形进行GrabCut分割
mask = foreground_background_segmentation_grabcut(image, scaled_rect)
# 使用GrabCut分割后的mask应用前景突出背景模糊效果
highlighted_image = apply_foreground_highlight_with_blur(image, mask)


def preprocess(image):
    eq_image = histogram_equalization(image)
    blurred = mean_filter(eq_image, 5)
    sharpened = unsharp_masking(blurred)
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# 显示原始图像及其直方图
show_image_and_histogram(image, 'Original Image')

# 直方图均衡化
equalized = histogram_equalization(image)
show_image_and_histogram(equalized, 'Equalized Image')
