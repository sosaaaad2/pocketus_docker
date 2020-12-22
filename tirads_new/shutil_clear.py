import cv2
import numpy as np
import os

def Water(img, left, right, top, bottom):
    #img = cv2.GaussianBlur(img,(3,3),0)
    center_x = int((left + right)/2)
    center_y = int((top + bottom)/2)
    width = right - left
    height = bottom - top
    new_left = max(int(left - width*0.8),0)
    new_right = min(int(right + width*0.8), img.shape[1])
    new_top = max(int(top - height*0.8),0)
    new_bottom = min(int(bottom + height*0.8), img.shape[0])

    #原本数据在新图片的坐标：
    left = left - new_left
    top = top - new_top

    mask_img = img[new_top:new_bottom, new_left:new_right]
    mask_img = cv2.cvtColor(mask_img,cv2.COLOR_GRAY2BGR)
    gary = cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)

    ret0, thresh0 = cv2.threshold(gary,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh0,cv2.MORPH_OPEN,kernel, iterations = 2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    #sure_bg[:,:] = 255
    #sure_bg[int(gary.shape[0]*0.2):int(gary.shape[0]*0.8), int(gary.shape[1]*0.2):int(gary.shape[1]*0.8)] = 0
    sure_bg[int(gary.shape[0]*0.85):int(gary.shape[0]*1), :] = 255
    sure_bg[:, int(gary.shape[1]*0.85):int(gary.shape[1]*1)] = 255
    sure_bg[0:int(gary.shape[0]*0.15), :] = 255
    sure_bg[0:int(gary.shape[0]*0.15), :] = 255

    # 确定前景区域
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret1, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0) #使用最大值0.7分割前景
    sure_fg_mask = np.zeros((gary.shape[0], gary.shape[1]))
    #sure_fg_mask[int(gary.shape[0]*0.2):int(gary.shape[0]*0.8), int(gary.shape[1]*0.2):int(gary.shape[1]*0.8)] = sure_fg[int(gary.shape[0]*0.2):int(gary.shape[0]*0.8), int(gary.shape[1]*0.2):int(gary.shape[1]*0.8)]
    sure_fg = sure_fg_mask
    sure_fg[int(top+height*0.2):int(top+height*0.8), int(left+width*0.2):int(left+width*0.8)] = 255

    # 查找未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # 标记标签
    ret2, markers1 = cv2.connectedComponents(sure_fg)
    markers = markers1+1
    markers[unknown==255] = 0

    markers3 = cv2.watershed(mask_img,markers)
    markers3[-3:-1, :] = 1
    markers3[:, -3:-1] = 1
    markers3[:, 0:3] = 1
    markers3[0:3, :] = 1

    mask_img[markers3 == -1] = [0,255,0]

    cv2.imshow('sure_fg', sure_fg)
    cv2.imshow('sure_bg', sure_bg)
    cv2.imshow('image2', mask_img)
    cv2.waitKey(0) 

    return mask_img, new_left, new_right, new_top, new_bottom

def sobel_demo(image):
    gray_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)  # x方向一阶导数
    gray_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)  # y方向一阶导数
    gradx = cv2.convertScaleAbs(gray_x)  # 转回原来的uint8形式
    grady = cv2.convertScaleAbs(gray_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)  # 图像融合
    return gradxy


# Scharr算子是Sobel算子的增强版本
def scharr_demo(image):
    grad_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    return gradxy

def lapalian_demo(image): #拉普拉斯算子
    dst = cv2.Laplacian(image, cv2.CV_32F)
    lpls = cv2.convertScaleAbs(dst)
    # 自己定义卷积核
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # dst = cv2.filter2D(image, cv2.CV_32F, kernel=kernel)
    # lpls = cv2.convertScaleAbs(dst) #单通道
    # cv2.imshow("lapalian", lpls)
    return lpls

def Collection(img, X, Y, sobelx, new_left, new_top, mask_img):
    # 对梯度图进行提取
    
    # 对边缘进行提取
    edge_mask = np.zeros(mask_imge.shape(0), mask_imge.shape(1))
    
    for i in range(X):
        e_x = X[i] - new_left
        e_y = Y[i] - new_top
        mask_img[e_x, e_y]
    return 1

def classify_clear(shape_meta, img, X, Y, zero_crossing, binary_shape, image_name, pred_cls, save=False):
    *_, contours, _ = cv2.findContours(binary_shape,
    cv2.RETR_EXTERNAL,  # 只取外围轮廓
    cv2.CHAIN_APPROX_NONE)  # cv2.CHAIN_APPROX_**决定得到的轮廓点的数目
    regular_temp_mask = r'D:\Users\XULANG197\Pictures\thyroid_data\shape_classification\results\regular'
    irregular_temp_mask = r'D:\Users\XULANG197\Pictures\thyroid_data\shape_classification\results\irregular'
    x,y,w,h = shape_meta['rect']
    edge_scale = shape_meta['zoom scale']
    x_change, y_change = shape_meta['xy_change']

    X = X/edge_scale + x_change
    Y = Y/edge_scale + y_change
    X = X.astype(np.int)
    Y = Y.astype(np.int)
    img_ori = img.copy()

    # 图片预处理
    img = cv2.equalizeHist(img)     #直方图均衡 
    img = cv2.blur(img,(3,3))    #进行滤波去掉噪声

    # 分水岭分割
    left = x
    right = x+w
    top = y
    bottom = y+h
    mask_img, new_left, new_right, new_top, new_bottom = Water(img, left, right, top, bottom)

    # 求解梯度
    sobelx=scharr_demo(img)
    if save:
        cv2.imshow("Blur", img) 
        cv2.imshow('img_ori', img_ori)
        cv2.imshow('sobel', sobelx)
        cv2.waitKey(0)

    if save:
        cv2.drawContours(img, contours, -1, (250,250,250), 1)
        for i in zero_crossing:
            cv2.circle(img, (X[i], Y[i]), 2, (220, 20, 120), 2)

        original_cnt = np.vstack((X, Y)).T
        if pred_cls == 0:
            save_ori_name = os.path.join(regular_temp_mask,image_name)
            save_name = os.path.join(regular_temp_mask,image_name[:-4] + '_c.jpg')
            cv2.imwrite(save_ori_name, img_ori)
            cv2.imwrite(save_name, img)
        else:
            save_ori_name = os.path.join(irregular_temp_mask,image_name)
            save_name = os.path.join(irregular_temp_mask,image_name[:-4] + '_c.jpg')
            cv2.imwrite(save_ori_name, img_ori)
            cv2.imwrite(save_name, img)

    return 1

