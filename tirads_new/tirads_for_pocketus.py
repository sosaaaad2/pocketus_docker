#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import math
import os
from itertools import combinations_with_replacement
import operator

import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
import sys

from tirads_new.shape_classification_configuration import REGULAR, IRREGULAR, cls_dict_ch, cls_dict_en, ELLIPSE, CIRCLE
from tirads_new.shutil_clear import *
from PIL import Image
from skimage import segmentation
from skimage.future import graph
from networkx.linalg import adj_matrix
from scipy.special import gamma
import time

class TiradsRecognition(object):
    def __init__(self, img, mask, is_debug=False):
        """
        类初始化方法，会自动调用初始化函数forward()
        :param img: 脱敏后的图
        :param mask: 脱敏后的mask
        :param is_debug: 是否要显示中间过程图像
        :raise: 当img为空时，raise Exception("the input img can not be None")
                当mask全黑时，raise Exception("the input mask can not be all black")
        """
        self.img = img
        self.mask = mask
        self.is_debug = is_debug
        self.mask_erode = None  # 被腐蚀后的mask
        self.box = None
        self.shadows = None  # 此图的声影mask
        self.cnt = None  # mask的countour
        self.box_point = None  # box的x, y, w, h。x、y是左上角

        # 初始化
        self.forward()


    def forward(self):
        """
        do some init operation.
        :raise: Exception(str)
        """
        if self.img is None:
            raise Exception("the input img can not be None")
        mask_copy = copy.deepcopy(self.mask)
        *_, contours, _ = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            raise Exception("the input mask can not be all black")

        sub_cnt_lens = [len(cnt) for cnt in contours]
        max_idx = np.argmax(sub_cnt_lens)
        self.cnt = contours[max_idx]

        self._get_box()

        self.mask_erode = self._erode(self.mask, 5)

    def _get_box(self):
        """
        get the box by the mask
        :return: numpy , the same shape as the mask
        """
        self.box = np.zeros(self.mask.shape, np.uint8)
        x, y, w, h = cv2.boundingRect(self.cnt)
        self.box_point = [x, y, w, h]
        self.box[y:y + h + 1, x:x + w + 1] = 255

###############################################################################
# 纵横比
###############################################################################
    def estimate_aspect_ratio(self, cal_type=3):
        """判断一个图像的纵横比，总共有三种方法。

        Args:
        ------
            :param cal_type:int,default 3
                cal_type 1：
                    根据两个轴与X轴之间的夹角判断，
                    与X夹角在[45,135]之间的轴为纵轴；
                cal_type 2：
                    根据在Y轴的投影判断,
                    在Y轴投影长度更长的为纵轴；
                cal_type 3：
                    根绝两个主轴在X与Y的投影长度比值判断，
                    轴1与轴2在Y轴的比值更长，说明轴1是纵轴，
                    在X轴的比值更长，说明轴1是横轴；
            :return estimated_ar:float
                            根据判断的纵横周计算的纵横比
            :return horizontal_axis:numpy.matrix.shape=(2,2)
                            纵轴
            :return vertical_axis:numpy.matrix.shape=(2,2)
                            横轴
        """
        img = copy.deepcopy(self.mask)
        (gx, gy), (ax1_len, ax1_ang), (ax2_len, ax2_ang) = self.main_components(img)
        ax1, ax2 = self.get_principal_axis(img)

        if cal_type == 1:
            if ax1_ang > 45 and ax1_ang < 135:
                estimated_ar = ax1_len / ax2_len
                estimated_ar = round(estimated_ar, 2)
                return estimated_ar, ax1, ax2
            else:
                estimated_ar = ax2_len / ax1_len
                estimated_ar = round(estimated_ar, 2)
                return estimated_ar, ax2, ax1

        elif cal_type == 2:
            # 通过三角函数来计算投影长度，其实直接用线的x，y坐标差计算更直接
            ax1_pj_len = np.sin(np.radians(ax1_ang)) * ax1_len
            ax2_pj_len = np.sin(np.radians(90 - ax1_ang)) * ax2_len

            if ax1_pj_len >= ax2_pj_len:
                estimated_ar = ax1_len / ax2_len
                estimated_ar = round(estimated_ar, 2)
                return estimated_ar, ax1, ax2
            elif ax1_pj_len < ax2_pj_len:
                estimated_ar = ax2_len / ax1_len
                estimated_ar = round(estimated_ar, 2)
                return estimated_ar, ax2, ax1

        elif cal_type == 3:
            x1, y1 = ax1[0, 0], ax1[1, 0]
            x2, y2 = ax1[0, 1], ax1[1, 1]
            x3, y3 = ax2[0, 0], ax2[1, 0]
            x4, y4 = ax2[0, 1], ax2[1, 1]

            ax1_pj_y, ax2_pj_y = abs(y1 - y2), abs(y4 - y3)
            ax1_pj_x, ax2_pj_x = abs(x1 - x2), abs(x4 - x3)

            pj_y = ax1_pj_y / (ax2_pj_y + 1e-4)
            pj_x = ax1_pj_x / (ax2_pj_x + 1e-4)

            if pj_y >= pj_x:  # ax1在Y方向更长(比值更大)，说明ax1应该是纵轴
                estimated_ar = ax1_len / ax2_len
                estimated_ar = round(estimated_ar, 2)
                return estimated_ar, ax1, ax2
            elif pj_y < pj_x:  # ax1在X方向更长（同理），说明ax1应该是横轴
                estimated_ar = ax2_len / ax1_len
                estimated_ar = round(estimated_ar, 2)
                return estimated_ar, ax2, ax1

    def main_components(self, shape):
        """ 利用协方差矩阵计算一个形状的质心，主轴长度以及角度。

        Args:
        ------
            :param shape:numpy.matrix
                输入形状，二值图

        Returns:
        ------
            :return (centroid_x, centroid_y):tuple
                                            质心点坐标
            :return (principal_axis1_len, principal_axis1_angle):(float, float)
                                            主轴1的长度、角度（与X轴正向）
            :return (principal_axis2_len, principal_axis2_angle):(float, float)
                                            主轴2的长度、角度（与X轴正向）

        Raise:
        ------
        TODO: to be added
        """

        binary_shape = copy.deepcopy(shape)
        principal_axis1, principal_axis2 = self.get_principal_axis(binary_shape)
        gx, gy = self.get_crossing(principal_axis1, principal_axis2)

        principal_axis1_len = self.get_axis_len(principal_axis1)
        principal_axis2_len = self.get_axis_len(principal_axis2)

        principal_axis1_angle, principal_axis2_angle = self.get_axis_angle(
            principal_axis1,
            principal_axis2)

        principal_axis1_angle = abs(principal_axis1_angle)
        principal_axis2_angle = abs(principal_axis2_angle)

        return (gx, gy), (principal_axis1_len, principal_axis1_angle), (principal_axis2_len, principal_axis2_angle)

    def get_axis_len(self, s1):
        p1 = np.asarray(s1[:, 0])
        p2 = np.asarray(s1[:, 1])

        ax_len = self.get_dist(p1, p2)
        ax_len = round(ax_len, 2)

        return ax_len

    def get_axis_angle(self, s1, s2):
        """计算一条直线与X轴的夹角。
        ang = atan((y2 - y1) / (x2 - x1))

        Args:
        -------
            :param s1: numpy.matrix.shape=(2,2)
                        直线1，通过两个点表示
                        [[x1, x2],
                        [y1, y2]]
            :param s2: numpy.matrix.shape=(2,2)
                        直线2，通过两个点表示
                        [[x3, x4],
                        [y3, y4]]

        Returns:
        -------
            :return ang1: float
                        s1与X轴夹角
            :return ang2: float
                        s2与X轴夹角
        """
        xa, ya = s1[0, 0], s1[1, 0]
        xb, yb = s1[0, 1], s1[1, 1]
        xc, yc = s2[0, 0], s2[1, 0]
        xd, yd = s2[0, 1], s2[1, 1]

        ang1 = math.degrees(math.atan((yb - ya) / (xb - xa + 1e-4)))
        ang2 = math.degrees(math.atan((yd - yc) / (xd - xc + 1e-4)))

        ang1 = round(ang1, 2)
        ang2 = round(ang2, 2)

        return abs(ang1), abs(ang2)

    def project_pt(self, point, line):
        """计算点投射在直线上的坐标，
        返回的坐标没有取整，因为不用于opencv绘图中，如果
        用于opencv绘图或者numpy索引记得取整。

        Args:
        ------
            :param point:list/numpy.array.shape=(2,)/tuple
                待计算点的(x, y)坐标，可以是以上形式
            :param line:numpy.matrix.shape=(2,2)
                [[x1, x2],
                [y1, y2]]
                投影的直线，以两个点的2X2矩阵的形式展现
        Returns:
        -------
            :return (px, py):tuple
                point投影在line上的xy坐标（浮点数）
        """
        x1 = line[0, 0]
        y1 = line[1, 0]
        x2 = line[0, 1]
        y2 = line[1, 1]
        x3, y3 = point[0], point[1]
        if x2-x1 == 0.0:
            return np.array([x1, y3])
        if y2-y1 == 0.0:
            return np.array([x3, y1])

        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        p_x = ((a * y3) + x3 - (a * b)) / (a ** 2 + 1)
        p_y = ((a ** 2 * y3) + (a * x3) + b) / (a ** 2 + 1)

        return np.array([p_x, p_y])

    def boundingRect(self, shape):
        # 利用opencv函数计算外接框（非最小外接框）
        # 未共用类cnt之前
        # gray_shape = copy.deepcopy(shape)
        # contours, _ = cv2.findContours(
        #     gray_shape,
        #     cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_NONE)
        # cnt = contours[0]

        return cv2.boundingRect(self.cnt)

    def hu_moment(self, binary_shape):
        """
        计算一个形状的Hu矩。

        Args:
        ------
            :param binary_shape:numpy.matrix
                输入形状，二值图
        Returns:
        -------
            :return M:dict
                输入形状的一阶二阶矩
        """
        binary_shape[binary_shape != 0] = 1
        nc_size, nr_size = np.shape(binary_shape)
        x_mgrid, y_mgrid = np.mgrid[0:nc_size, 0:nr_size]

        m00 = np.sum(binary_shape)
        if m00 == 0:
            print("The mask is empty!")
            return None

        m10 = np.sum(x_mgrid * binary_shape)
        m01 = np.sum(y_mgrid * binary_shape)

        xmean = m10 / m00
        ymean = m01 / m00

        # normalized center moment, in opencv moments it is called mu for short
        cm00 = m00
        cm02 = (np.sum(((y_mgrid - ymean) ** 2) * binary_shape)) / (m00 ** 2)
        cm11 = (np.sum((x_mgrid - xmean) * (y_mgrid - ymean) * binary_shape)) / (m00 ** 2)
        cm20 = (np.sum(((x_mgrid - xmean) ** 2) * binary_shape)) / (m00 ** 2)

        M = {'m00': m00,
             'm01': m01,
             'm10': m10,
             'mu00': cm00,
             'mu02': cm02,
             'mu11': cm11,
             'mu20': cm20}

        return M

    def get_centroid(self, binary_shape):
        """利用一阶矩计算形状的质心坐标。

        Args:
        ------
            :param binary_shape:numpy.matrix
                输入形状，二值图
        Returns:
        -------
            :return (cx, cy):tuple
                输入形状的质心坐标

        """
        # 未共用类cnt
        # def get_centroid(self, binary_shape):
        # shape = copy.deepcopy(binary_shape)
        # contours, _ = cv2.findContours(
        #     shape,
        #     cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_NONE)
        # mx_len_cnt_idx = np.argmax([len(i) for i in contours])
        # cnt = contours[mx_len_cnt_idx]
        # cnt = cnt.squeeze()

        M = cv2.moments(self.cnt)  # opencv based moment calculation
        # M = self.cal_moments(binary_shape)    #self implemented moment calculator

        cx = int(M['m10'] / M['m00'])  # centroid x  (in numpy array, is the column)
        cy = int(M['m01'] / M['m00'])  # centroid y  (in numpy array, is the row)

        return (cx, cy)

    def get_crossing(self, s1, s2):
        """
        寻找两条直线的交点坐标，在形状中，两条主轴的交点即质心。
        这个方法能够实现获得质心的功能。

        Args:
        -------
            :param s1: numpy.matrix.shape=(2,2)
                        直线1，通过两个点表示
                        [[x1, x2],
                        [y1, y2]]
            :param s2: numpy.matrix.shape=(2,2)
                        直线2，通过两个点表示
                        [[x3, x4],
                        [y3, y4]]

        Returns:
        -------
            :return (x, y): int
                centorid coordinate
        """
        xa, ya = s1[0, 0], s1[1, 0]
        xb, yb = s1[0, 1], s1[1, 1]
        xc, yc = s2[0, 0], s2[1, 0]
        xd, yd = s2[0, 1], s2[1, 1]

        a = np.matrix(
            [
                [xb - xa, -(xd - xc)],
                [yb - ya, -(yd - yc)]
            ])

        delta = np.linalg.det(a)

        # 不相交
        if np.fabs(delta) < 1e-6:
            print('Not crossing each other!')
            return None

        c = np.matrix(
            [
                [xc - xa, -(xd - xc)],
                [yc - ya, -(yd - yc)]
            ])

        d = np.matrix(
            [
                [xb - xa, xc - xa],
                [yb - ya, yc - ya]
            ])

        lamb = np.linalg.det(c) / delta
        miu = np.linalg.det(d) / delta

        if lamb <= 1 and lamb >= 0 and miu >= 0 and miu <= 1:
            x = int(xc + miu * (xd - xc))
            y = int(yc + miu * (yd - yc))
            return (x, y)

    def get_true_axis(self, shape, s1):
        """
        获得实际长度的主轴表达式（最小外接框长度）
        将轮廓点分别投影在轴上，取两端端点，作为实际轴
        的表达式。
         Args:
        ------
            :param shape:numpy.matrix
                输入形状，二值图
            :param s1:numpy.matrix.shape=(2,2)
                        原主轴，通过两个点表示
                        [[xa, xb],
                        [ya, yb]]
        Returns:
        -------
            :return s1:numpy.matrix.shape=(2,2)
                        实际主轴长度，通过两个点表示，
                        两个点分别表示形状在主轴上的最远点
                        [[xc, xd],
                        [yc, yd]]
        """
        xa, ya = s1[0, 0], s1[1, 0]
        xb, yb = s1[0, 1], s1[1, 1]

        l = self.get_dist((xa, ya), (xb, yb))
        x_index, y_index = np.linspace(xa, xb, l), np.linspace(ya, yb, l)

        xx = np.where(x_index.astype(np.int) < shape.shape[1])
        yy = np.where(y_index.astype(np.int) < shape.shape[0])
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        if xx.size == x_index.size and yy.size == y_index.size:
            pass
        elif xx.size < yy.size:
            x_index = x_index[xx]
            y_index = y_index[xx]
        else:
            x_index = x_index[yy]
            y_index = y_index[yy]

        binary_shape = copy.deepcopy(shape)
        # contours, _ = cv2.findContours(
        #     binary_shape,
        #     cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_NONE)
        # # 190809更新，动态选择最长的轮廓，为外轮廓
        # mx_len_cnt_idx = np.argmax([len(i) for i in contours])
        # cnt = contours[mx_len_cnt_idx]
        # cnt = cnt.squeeze()

        _project_pts = []
        # print(self.cnt.shape, self.cnt.squeeze().shape)
        for p in self.cnt.squeeze():
            pp = self.project_pt(p, s1)
            _project_pts.append(pp)

        # 0814修改， 垂直线段取到的结果可能是错的，在原始确认基础上，判断是否是垂直
        project_pts = np.matrix(_project_pts)
        project_pts = project_pts[project_pts.argsort(axis=0)[:, 0]]  # x轴排序，从小到大
        project_pts = project_pts.squeeze().tolist()

        axis_len = self.get_dist(project_pts[-1], project_pts[0])
        start, end = project_pts[0], project_pts[-1]
        # 如果是完全垂直的话，上面这样子排序是不管用的
        if start[0] == end[0]:  # x相等，垂直, 要从y取
            # print('same X!')
            project_pts = np.matrix(_project_pts)
            start_idx, end_idx = np.where(project_pts[:,1]==np.min(project_pts[:,1]))[0][0], \
                                 np.where(project_pts[:,1]==np.max(project_pts[:,1]))[0][0]
            # print(start_idx, end_idx, project_pts[start_idx], project_pts[end_idx])
            project_pts = project_pts.squeeze().tolist()
            # print(project_pts[start_idx], project_pts[end_idx])
            start, end = project_pts[start_idx], project_pts[end_idx]
            # print([start, end])
        # s1 = [start, end]
        # s1 = np.matrix(s1).T
        #
        # axis_len = self.get_dist(project_pts[-1], project_pts[0])
        # start, end = project_pts[0], project_pts[-1]
        s1 = [start, end]
        s1 = np.matrix(s1).T

        return s1

    def get_principal_axis(self, shape, return_angle=False):
        """通过协方差矩阵计算一个形状的两条主轴（以两点表示一条直线，具体排列方式下面有描述）

        Args:
        ------
            :param shape:numpy.matrix
                输入形状，二值图
            :param return_angle:bool, default False
                是否返回主轴theta角度

        Returns:
        -------
            :return principal_axis1:numpy.matrix.shape=(2,2)
            :return principa2_axis1:numpy.matrix.shape=(2,2)

        Raise:
        ------
        TODO: to be added
        """
        binary_shape = copy.deepcopy(shape)
        y, x = np.nonzero(binary_shape)

        # x = x - np.mean(x)
        # y = y - np.mean(y)
        coords = np.vstack([x, y])

        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)

        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with the largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]

        # 确定两条主轴的实际长度
        # 先画一条确定超过形状长度的轴
        # 再根据投影点找到实际距离
        _, _, w, h = self.boundingRect(binary_shape)
        scale = np.max((w, h)) * .4

        principal_axis1 = np.matrix(
            [
                [x_v1 * -scale * 2 + np.mean(x), x_v1 * scale * 2 + np.mean(x)],
                [y_v1 * -scale * 2 + np.mean(y), y_v1 * scale * 2 + np.mean(y)]
            ])
        principal_axis2 = np.matrix(
            [
                [x_v2 * -scale * 2 + np.mean(x), x_v2 * scale * 2 + np.mean(x)],
                [y_v2 * -scale * 2 + np.mean(y), y_v2 * scale * 2 + np.mean(y)]
            ])

        # 真实长度（最小外接框长度）
        principal_axis1 = self.get_true_axis(binary_shape, principal_axis1)
        principal_axis2 = self.get_true_axis(binary_shape, principal_axis2)

        if return_angle:
            theta = np.tanh((x_v1) / (y_v1))
            theta = theta * 180 / np.pi
            return principal_axis1, principal_axis2, theta
        else:
            return principal_axis1, principal_axis2

    def get_dist(self, pt1, pt2):
        dist = math.sqrt(math.pow((pt1[0] - pt2[0]), 2) + math.pow((pt1[1] - pt2[1]), 2))
        dist = round(dist, 2)
        return dist

###############################################################################
# 形状分类
###############################################################################
    def classify_shape(self, true_cls=None, ZERO=2e-2, save=False, test_result_save_path=None, filename=None, cal_type=1):
        """
        判断形状为规则形状还是不规则形状的分类算法。
        :param binary_shape: 输入原始形状图像，numpy矩阵
        :param true_cls: 实际标签，如果有的话给出，REGULAR 或 IRREGULAR，没有的话不给，则为推断模式
        :param ZERO: 寻找零值点的阈值，由于形状的曲率不一定刚好有0点，规定绝对值在这个数值以内的点为零点
        :param save: 是否保存中间结果，分析结果用，bool，默认False
        :param test_result_save_path: 结果的保存路径，str,默认None
        :param filename: 保存文件名，str,默认None
        :param cal_type:计算方式，int，0为常规，基于各部分突起判断；1在0的基础上多增一个曲率判断，纠正可能被误判为规则的不规则形状
        :return final_class:预测类别，str，规则或不规则

        :raise:
            ValueError
            ZeroDivisionError
        """
        binary_shape = self.mask
        # 形状尺度归一化， 将长轴归一化至128大小，由于四方各留5个像素，实际上是118
        zoom_shape, shape_meta, [left_edge,right_edge,top_edge,bottom_edge] = self.scale_normalize(binary_shape, L=128)   # 长轴归一化
        zoom_shape = self.smooth_boundary(zoom_shape)                     # 边界平滑
        zoom_scale = shape_meta['zoom scale']
        h, w = zoom_shape.shape
        cnt = self.get_contour(zoom_shape)
        x, y = cnt[:, 0], cnt[:, 1]
        original_sigma = 2.33
        if zoom_scale >= 1.5:
            sigma_scale = zoom_scale * .5
        elif zoom_scale > 1 and zoom_scale < 1.5:
            sigma_scale = zoom_scale
        else:
            sigma_scale = 1
        sigma, range_ = sigma_scale * original_sigma, math.ceil(sigma_scale * original_sigma * 3)

        # 对轮廓坐标点作做高斯平滑
        curvature_calculator = CurvatureCalculator()
        X = curvature_calculator.gaussian_smooth(x, sigma=sigma, range_=range_, plot=False)
        Y = curvature_calculator.gaussian_smooth(y, sigma=sigma, range_=range_, plot=False)
        smoothed_cnt = np.vstack((X, Y)).T

        # # --- plot 检视平滑效果的 ----
        save = False
        edge_if = True   # 是否进行边缘相切判断
        # if save:
        #     f, a = plt.subplots(1, 3, figsize=(3 * 5 * w / 128, 1 * 4 * h / 128))

        #     a[0].imshow(zoom_shape, cmap=plt.cm.Greys_r)
        #     a[0].set_axis_off()
        #     #a[0].set_title('类别：{0}'.format(cls_dict_ch[true_cls]))
        #     a[0].set_title('类别：{0}')

        #     a[1].plot(x, y, 'b-')
        #     a[1].set_title('未平滑轮廓')
        #     a[1].yaxis.set_ticks_position('left')  # 将y轴的位置设置在右边
        #     a[1].invert_yaxis()  # y轴反向

        #     a[2].plot(X, Y, 'b-')
        #     a[2].set_title('平滑后轮廓 sigma:{0:.2f} miu:{1}'.format(sigma, range_))
        #     a[2].yaxis.set_ticks_position('left')  # 将y轴的位置设置在右边
        #     a[2].invert_yaxis()  # y轴反向
        # # -------end plot------

        # 根据平滑后的cnt计算曲率
        # 0814修改，直接设置为长度的0.1， 不做与15个pixels比较去更小的
        N = min(15, int(len(cnt) * .1))
        #N = int(len(cnt) * .1)
        cv = curvature_calculator.curvature(smoothed_cnt, N=N)
        ZERO = 0.025
        zero_crossing = np.where(abs(cv) < (ZERO))[0]
        zero_crossing = np.hstack((zero_crossing, curvature_calculator.find_zero_crossings(cv)))
        zero_crossing = np.array(list(set(zero_crossing)))

        # 不筛查过零点，逐一遍历
        # 点找足了是正确判断的必要条件
        zero_crossing.sort()
        zero_crossing = zero_crossing.astype(np.int)
        if save:
            f1, (a11, a12) = plt.subplots(1, 2,
                                          figsize=(w / 128 * 5 + 5, h / 128 * 4),
                                          gridspec_kw={'width_ratios': [w / 128, 1], 'height_ratios': [h / 128, ]})
            a11.plot(X, Y, 'b-')
            a11.invert_yaxis()
            a12.plot(cv)
            a12.axhline(y=0, color='r')
        
            a11.plot(X[zero_crossing], Y[zero_crossing], 'rx')
            a12.plot(zero_crossing, cv[zero_crossing], 'rx')
            a12.set_title('过零点（小于{0}）'.format(ZERO))


        
        # 寻找曲率过0点，可能会找出几个连续的过零点，可能会找不到刚好为0的点
        # 解决：找绝对值小于ZERO的点作为0点，以及前后数值符号不同的点，不筛除连续的过0点
        # 两两一对过零点，找他们之间的突起（曲率局大值）
        # 以突起的点的下标作为标识，避免重复计算同一个突起

        triangle_left, triangle_max, triangle_right = self.find_zero_points(zero_crossing, cv, cnt, ZERO)
        if len(triangle_max) == 0 or len(triangle_left) == 0:
            return REGULAR

        # 寻找曲率凹陷最大点
        cv_turn = cv * -1
        _, triangle_min, _ = self.find_zero_points(zero_crossing, cv_turn, cnt, ZERO)

        # 极值过滤  找到最大的两个极值。并过滤掉较小的极值
        max_thresh = -0.041
        triangle_max_temp = cv[triangle_max]
        triangle_max_temp.sort()
        second_max = triangle_max_temp[1]
        max_thresh = max(max_thresh, second_max*.85)
        for i, max_point in enumerate(triangle_max):
            if cv[max_point] > max_thresh:
                triangle_left[i] = -1 
                triangle_max[i] = -1
                triangle_right[i] = -1
        while -1 in triangle_left:
            triangle_left.remove(-1)
        while -1 in triangle_max:
            triangle_max.remove(-1)
        while -1 in triangle_right:
            triangle_right.remove(-1)

        # ------plot 筛选后图点-------
        if save:
            f2, (a21, a22) = plt.subplots(1, 2,
                                          figsize=(w / 128 * 5 + 5, h / 128 * 4),
                                          gridspec_kw={'width_ratios': [w / 128, 1], 'height_ratios': [h / 128, ]})
            a21.plot(X, Y, 'b-')
            a21.invert_yaxis()
            a22.plot(cv)
            a22.axhline(y=0, color='r')
            a22.set_title('筛选后')
        # ----- end plot------

        # -----开始计算每一段凸起的性质------
        count = 0
        shape_area = np.count_nonzero(zoom_shape)
        shape_perimeter = cv2.arcLength(cnt, True)

        txt = '形状面积：{0} 周长：{1:.2f}'.format(np.count_nonzero(zoom_shape), shape_perimeter)
        table = np.zeros(shape=(len(triangle_left), 4))
        
        for left_pt_idx, vertex, right_pt_idx in zip(triangle_left[:-1], triangle_max[:-1], triangle_right[:-1]):
            # # ---- plot -----
            if edge_if == True:
                if smoothed_cnt[vertex][0] < left_edge or smoothed_cnt[vertex][0] > right_edge \
                    or smoothed_cnt[vertex][1] < top_edge or smoothed_cnt[vertex][1] > bottom_edge:
                    continue
            if save:
                a21.plot(X[left_pt_idx], Y[left_pt_idx], 'rx')
                a21.text(X[left_pt_idx], Y[left_pt_idx], str(left_pt_idx))
            
                a21.plot(X[vertex], Y[vertex], 'yx')
                a21.text(X[vertex], Y[vertex], str(count))
                a21.plot(X[right_pt_idx], Y[right_pt_idx], 'rx')
                a21.text(X[right_pt_idx], Y[right_pt_idx], str(right_pt_idx))
            
                a22.plot(left_pt_idx, cv[left_pt_idx], 'rx')
                a22.plot(vertex, cv[vertex], 'yx')
                a22.text(vertex, cv[vertex], str(count))
                a22.plot(right_pt_idx, cv[right_pt_idx], 'rx')
            # # ---- end plot -----

            try:
                angle = self.point_angle(smoothed_cnt[left_pt_idx], smoothed_cnt[vertex], smoothed_cnt[right_pt_idx])
                angle = round(angle, 2)
            except ZeroDivisionError:
                # 一般说明三个点基本共线了，并且也是离得比较近
                angle = 180

            arc_length = cv2.arcLength(cnt[left_pt_idx:right_pt_idx], False)
            arc_length = round(arc_length, 2)
            euclidean = self.get_dist(smoothed_cnt[left_pt_idx], smoothed_cnt[right_pt_idx])
            arc_area = self.create_shape(cnt[left_pt_idx:right_pt_idx],
                                         output_max=zoom_shape.max(),
                                         shape=zoom_shape.shape,
                                         dtype=zoom_shape.dtype)
            arc_area = np.count_nonzero(arc_area)  # 这段突起的弧长面积
            table[count, 0] = angle
            table[count, 1] = arc_length
            table[count, 2] = euclidean
            table[count, 3] = arc_area
            description = 'NO.{0} ({1} {2} {3}) 角度：{4} 弧长：{5} 欧式：{6} 弧面积：{7}'.format(count,
                                                                                     left_pt_idx, vertex, right_pt_idx,
                                                                                     angle, arc_length, euclidean,
                                                                                     arc_area)
            txt = txt + '\n' + description
            count += 1

        # 最后一个点
        left_pt_idx, vertex, right_pt_idx = triangle_left[-1], triangle_max[-1], triangle_right[-1]
        vertex = vertex if vertex < len(cnt) else vertex % len(cnt)
        edge_if_last = smoothed_cnt[vertex][0] < left_edge or smoothed_cnt[vertex][0] > right_edge \
            or smoothed_cnt[vertex][1] < top_edge or smoothed_cnt[vertex][1] > bottom_edge
        if edge_if and edge_if_last:
            aaaa = 111
            angle = 180
        else:    
            # ---- plot ----
            if save:
                a21.plot(X[left_pt_idx], Y[left_pt_idx], 'rx')
                a21.text(X[left_pt_idx], Y[left_pt_idx], str(left_pt_idx))
            
                a21.plot(X[vertex], Y[vertex], 'yx')
                a21.text(X[vertex], Y[vertex], str(count))
                a21.plot(X[right_pt_idx], Y[right_pt_idx], 'rx')
                a21.text(X[right_pt_idx], Y[right_pt_idx], str(right_pt_idx))
            
                a22.plot(left_pt_idx, cv[left_pt_idx], 'rx')
                a22.plot(vertex, cv[vertex], 'yx')
                a22.text(vertex, cv[vertex], str(count))
                a22.plot(right_pt_idx, cv[right_pt_idx], 'rx')
            # ---- end plot ----

            try:
                angle = self.point_angle(smoothed_cnt[left_pt_idx], smoothed_cnt[vertex], smoothed_cnt[right_pt_idx])
                angle = round(angle, 2)
            except ZeroDivisionError:
                # 一般说明有两条边特别接近
                angle = 180
            if right_pt_idx < left_pt_idx:
                arc_cnt = np.vstack((cnt[left_pt_idx:], cnt[:right_pt_idx]))
                arc_length = cv2.arcLength(arc_cnt, False)
            else:
                arc_cnt = cnt[left_pt_idx:right_pt_idx]
                arc_length = cv2.arcLength(arc_cnt, False)
            arc_length = round(arc_length, 2)
            euclidean = self.get_dist(smoothed_cnt[left_pt_idx], smoothed_cnt[right_pt_idx])   # 角左右两点面积
            arc_area = self.create_shape(arc_cnt,
                                        output_max=zoom_shape.max(),
                                        shape=zoom_shape.shape,
                                        dtype=zoom_shape.dtype)
            arc_area = np.count_nonzero(arc_area)  # 这段突起的弧长面积
            table[count, 0] = angle
            table[count, 1] = arc_length
            table[count, 2] = euclidean
            table[count, 3] = arc_area
            description = 'NO.{0} ({1} {2} {3}) 角度：{4} 弧长：{5} 欧式：{6} 弧面积：{7}'.format(count,
                                                                                 left_pt_idx, vertex, right_pt_idx,
                                                                                 angle, arc_length, euclidean, arc_area)
            txt = txt + '\n' + description
        count += 1

        # ----判断------
        sharp_angle_idx = np.where((table[:, 0] <= 110.0) & (table[:, 0] > 10))[0]
        if len(sharp_angle_idx) == 0:  # 无定义上的锐角，说明是规则形状
            pred_cls = REGULAR
        else:
            if len(sharp_angle_idx) == 1:  # 有一个锐角，如果这个角特别大，可能是一个规则形状的误判角度
                euclidean = table[sharp_angle_idx, 2]
                arc_area = table[sharp_angle_idx, 3]
                if arc_area == np.max(table[:, 3]):
                    pred_cls = REGULAR
                else:
                    if arc_area / euclidean >= 10 or arc_area / shape_area > .1:
                        pred_cls = REGULAR
                    else:
                        pred_cls = IRREGULAR
            elif len(sharp_angle_idx) > 1:
                for i in sharp_angle_idx:
                    euclidean = table[i, 2]
                    arc_area = table[i, 3]
                    angel_left = table[i-1, 0]
                    angel_right = table[i-len(table)+1, 0]
                    if arc_area == np.max(table[:, 3]): continue
                    if arc_area / euclidean >= 10 or arc_area / shape_area > .1:
                        pred_cls = REGULAR
                    elif table[i, 0]>100 and table[i-1, 0]>105 and table[i-len(table)+1, 0]>105:
                        pred_cls = REGULAR
                    else:
                        pred_cls = IRREGULAR
                        break
            else:
                pass

        # 以下操作不应该与上面的角度判定操作起到相反作用，不能给影响到上面已经成型的正确判断
        # 这部分还有一定的不足之处，由于只看曲率的极值，某些仅在一处产生较大曲率的形状，也有可能产生误判
        # 通过曲率来判断是一个较可靠的方法，但是仍要思考用更全面的方法考虑曲率
        # 纠正不规则形状被误判为规则的：
        if cal_type == 1:   # 是否在上面的基础上继续基于曲率进行判断
            if pred_cls == REGULAR and cv.max() - cv.min() >= .2:
                if len(sharp_angle_idx) <= 1:
                    if cv.max() / (cv.max() - cv.min()) > .5 or abs(cv.min() / (cv.max() - cv.min())) > .92 or abs(
                            cv.min()) > 0.21:
                        pred_cls = IRREGULAR
                else:  # 有多个锐角还被判断为规则的形状稍微降低一点要求
                    if cv.max() / (cv.max() - cv.min()) > .5 or abs(cv.min() / (cv.max() - cv.min())) > .92:
                        pred_cls = IRREGULAR
        # 判断是否有曲率为负的尖锐锐角
        for p_min in triangle_min:
            cv_min = cv[p_min]
            p_min_left = 0
            p_min_right = len(cv) - 1
            temp = triangle_max.copy()
            temp.sort()
            if temp[0] > p_min:  # 该点位于所有极值点左侧
                p_min_left = temp[-1]
                p_min_right = temp[0]
                gradient_left = (cv_min - cv[p_min_left])/(p_min + len(cnt) - p_min_left)
                gradient_right = (cv_min - cv[p_min_right])/(p_min_right - p_min)
                range_left = cv_min - cv[p_min_left]
                range_right = cv_min - cv[p_min_right]
            elif p_min > temp[-1]:# 该点位于所有极值点右侧
                p_min_left = temp[-1]
                p_min_right = temp[0]
                gradient_left = (cv_min - cv[p_min_left])/(p_min - p_min_left)
                gradient_right = (cv_min - cv[p_min_right])/(p_min_right + len(cnt) - p_min)
                range_left = cv_min - cv[p_min_left]
                range_right = cv_min - cv[p_min_right]
            else:
                for p_max in triangle_max:
                    if abs(p_max - p_min) < abs(p_min_left - p_min) and p_max - p_min < 0:
                        p_min_left = p_max
                for p_max in triangle_max:
                    if abs(p_max - p_min) < abs(p_min_right - p_min) and p_max - p_min > 0:
                        p_min_right = p_max
                gradient_left = (cv_min - cv[p_min_left])/(p_min - p_min_left)
                gradient_right = (cv_min - cv[p_min_right])/(p_min_right - p_min)
                range_left = cv_min - cv[p_min_left]
                range_right = cv_min - cv[p_min_right]
            # 金标准版本
            if cv_min > 0.04:
                if (gradient_left * range_left > 0.00084) or cv_min > 0.1 or \
                    (gradient_left > 0.0045 and range_left > 0.11):
                    pred_cls = IRREGULAR
                if (gradient_right * range_right > 0.00084) or cv_min > 0.1 or \
                    (gradient_right > 0.0045 and range_right > 0.11):
                    pred_cls = IRREGULAR
            #分割算法版本
            # if cv_min > 0.025:
            #     if (gradient_left * range_left > 0.00034) or cv_min > 0.1 or \
            #         (gradient_left > 0.0045 and range_left > 0.11):
            #         pred_cls = IRREGULAR
            #     if (gradient_right * range_right > 0.00034) or cv_min > 0.1 or \
            #         (gradient_right > 0.0045 and range_right > 0.11):
            #         pred_cls = IRREGULAR
        # 对于锐角，如果两边是负角度，且负角度够大
        for i in range(len(table)): 
            if table[i,0] < 90 and table[i,0] > 10:
                all_point = triangle_min + triangle_max
                all_point.sort()
                loca = all_point.index(triangle_max[i])
                p_max_left = all_point[loca-1]
                p_max_right = all_point[(loca+1)%len(all_point)]
                if p_max_left in triangle_min and p_max_right in triangle_min:
                    if (cv[p_max_left] > 0.05 and cv[p_max_right] > 0.05) or \
                        (cv[p_max_left] > 0.025 and cv[p_max_right] > 0.025 and table[i,0] < 60):
                        pred_cls = IRREGULAR
                # if p_max_left in triangle_min or p_max_right in triangle_min:
                #     if (cv[p_max_left] > 0.05 or cv[p_max_right] > 0.05) or \
                #         (cv[p_max_left] > 0.025 or cv[p_max_right] > 0.025 and table[i,0] < 70):
                #         pred_cls = IRREGULAR
                # if table[i,0] < 70:
                #     pred_cls = IRREGULAR
          
        # 纠正规则形状误判为不规则的
        # TODO

        # if save:
        #     a21.set_title('类别：{0} 预测：{1}'.format(cls_dict_ch[true_cls], cls_dict_ch[pred_cls]))
        #     assert test_result_save_path is not None, 'Incorrect save path!'
        #     assert filename is not None, 'Incorrect save filename!'
        #     save_path = os.path.join(test_result_save_path, cls_dict_en[true_cls])
        #     if not os.path.exists(save_path): os.mkdir(save_path)
        #     if true_cls == pred_cls:
        #         save_path = os.path.join(save_path, 'right')
        #     else:
        #         save_path = os.path.join(save_path, 'wrong')
        #     if not os.path.exists(save_path): os.mkdir(save_path)
        #
        #     filename_suffix = filename.split('.')[-1]  # 如果输入文件名内不包含类似“.jpg”的后缀的话，保存结果可能会出错
        #     filename_suffix = '.' + filename_suffix
        #     f.savefig(os.path.join(save_path, filename.replace(filename_suffix, '_1.png')),
        #               format='png',
        #               transparent=True,
        #               dpi=300,
        #               pad_inches=0,
        #               bbox_inches='tight')
        #
        #     f1.savefig(os.path.join(save_path, filename.replace(filename_suffix, '_2.png')),
        #                format='png',
        #                transparent=True,
        #                dpi=300,
        #                pad_inches=0,
        #                bbox_inches='tight')
        #
        #     f2.savefig(os.path.join(save_path, filename.replace(filename_suffix, '_3.png')),
        #                format='png',
        #                transparent=True,
        #                dpi=300,
        #                pad_inches=0,
        #                bbox_inches='tight')
        #
        #     txt_save_path = os.path.join(save_path, filename.replace(filename_suffix, '_t.txt'))
        #     with open(txt_save_path, 'w') as f:
        #         f.write(txt)
        #     plt.close('all')
        # else:
        #     pass
        #     # print(txt)
        ############# 进行清晰模糊判断  ###########################
        # clear_ans = classify_clear(shape_meta, self.img, X, Y, zero_crossing, binary_shape, self.image_name, pred_cls, save=True)
        return pred_cls


    def find_zero_points(self,zero_crossing, cv, cnt, ZERO):
        # 寻找曲率过0点，可能会找出几个连续的过零点，可能会找不到刚好为0的点
        # 解决：找绝对值小于ZERO的点作为0点，以及前后数值符号不同的点，不筛除连续的过0点
        # 两两一对过零点，找他们之间的突起（曲率局大值）
        # 以突起的点的下标作为标识，避免重复计算同一个突起
        count = 0
        num_zero_crossing = len(zero_crossing)   # 过零点对应坐标
        triangle_left = []
        triangle_max = []
        for i in range(num_zero_crossing + 1):
            for next_i in range(i + 1, num_zero_crossing + 1):
                if next_i >= num_zero_crossing:
                    next_i = 0
                    interval = np.hstack((cv[zero_crossing[i] + 1:], cv[:zero_crossing[0]+1]))
                else:
                    interval = cv[zero_crossing[i] + 1:zero_crossing[next_i % num_zero_crossing]]  # 取过零点对应的一段
                try:
                    abs_max_idx = np.argmax(abs(interval))
                except ValueError:
                    continue  # 相邻过零点之间是空序列

                # 区间是朝上（凹的）的也不看，可以跳过
                if np.all(abs(interval[1:]) < ZERO): continue  # 这一段都是零点
                if interval[abs_max_idx] > ZERO: break  # 朝上（凹点）' 最大值大于ZERO。说明是朝上点   
                #             if abs(interval[abs_max_idx]) <= ZERO: continue  # 去除连续0点的影响
                # 确认next_i:
                abs_max_cnt_idx = np.argmin(interval)
                abs_max_cnt_idx = zero_crossing[i] + abs_max_cnt_idx
                # 最后一个点和第一个点之间的情况
                abs_max_cnt_idx = abs_max_cnt_idx if abs_max_cnt_idx < len(cnt) else abs_max_cnt_idx % len(cnt)
                if abs_max_cnt_idx in triangle_max: break  # 以这个点为凸点已经计算过了，skip                
                #加入修正，使左点变成绝对值最小的点
                # try:
                #     left_zero = cv[zero_crossing[i]: zero_crossing[next_i-1]]
                #     left_zero = abs(left_zero)
                #     left_zero[left_zero < 0.001] = 0
                #     left_zero = np.flip(left_zero, axis=0)
                #     left_pt_idx = len(left_zero) - np.argmin(left_zero) - 1 + zero_crossing[i]

                left_pt_idx, right_pt_idx = zero_crossing[i], zero_crossing[next_i]
                triangle_max.append(abs_max_cnt_idx)
                triangle_left.append(left_pt_idx)
                # 这个时候要重新选择左右点，要向左以及向右选择最原理凸点的点
                # 左边的点（从下标的角度出发）可以不用再找，因为我们已经确认他是原理凸点的那个点了
                # 右边的点需要重确认：需要确认目前的right_pt_idx的下一个点不是非零点，如果是，就要换非零点了，
                count += 1
                break

        # 确定所有凸点以及对应左边的端点，接下来要找到其最右边的端点
        # 为何最右，要确保两边的端点都是最远离凸点的第段曲率过零点
        triangle_right = []
        if len(triangle_max) == 0 or len(triangle_left) == 0:
            return triangle_left, triangle_max, triangle_right
        for vertex, left_pt_idx in zip(triangle_max[:-1], triangle_left[:-1]):
            for i in range(1, num_zero_crossing): # 找到vertex之后的归零点对应的i，然后跳出循环
                if zero_crossing[i] > vertex:   
                    break
            right_candidate_arr = zero_crossing[i:]
            right_pt_idx = right_candidate_arr[0]
            for ri in right_candidate_arr[1:]:  # 保证连续。找到右侧归零点最不连续的一个
                if ri - right_pt_idx <= 1: # and abs(cv[ri]) - abs(cv[right_pt_idx]) < 0:
                    right_pt_idx = ri
                else:  # 不连续就可以跑了
                    triangle_right.append(right_pt_idx)
                    break
        if len(triangle_right) == len(triangle_left[:-1]) - 1:
            triangle_right.append(right_pt_idx)

        # ----最后一个点-----
        vertex, left_pt_idx = triangle_max[-1], triangle_left[-1]
        right_start = np.where(zero_crossing > vertex)[0]
        if len(right_start) == 0:
            right_candidate_arr = zero_crossing
        else:
            right_candidate_arr = np.array(list(zero_crossing[right_start[0]:]) + list(zero_crossing))
        right_pt_idx = right_candidate_arr[0]
        for j, ri in enumerate(right_candidate_arr[1:]):  # 保证连续
        #if abs(cv[ri]) - abs(cv[right_pt_idx]) < 0:
            if (ri - right_pt_idx <= 1 and ri - right_pt_idx >= 0) or (
                    ri == zero_crossing[0] and right_candidate_arr[j + 2] - ri == 1):
                right_pt_idx = ri
            else:
                triangle_right.append(right_pt_idx)
                break
        if len(triangle_right) < len(triangle_left):
            triangle_right.append(right_pt_idx)

        return triangle_left, triangle_max, triangle_right
##############################################################################
# 彗星尾
##############################################################################
    def find_comet_calcify(self, mode, cys_mask):
        """
        彗星尾检测算法
        :param mode: report or find, str类型
        :return: if mode == "report": 彗星尾数量: int
                 if mode == "find":  comet_ridge_map: numpy array
                                    返回彗星尾的ridge垂线图,  如果没有彗星尾，则会返回0
        """
        img = copy.deepcopy(self.img)
        mask = copy.deepcopy(self.mask)

        if self.is_debug:
            cv2.imshow("ori_img", img)
            cv2.imshow("ori_mask", mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_erode = cv2.erode(mask, kernel)

        dst = self.set_zero_mask(img, mask, 20)

        # 将所有输入都裁剪成box的区域
        xmin, ymin, w, h = self.box_point
        xmax = xmin + w
        ymax = ymin + h
        dst = dst[ymin:ymax + 1, xmin:xmax + 1]
        mask = mask[ymin:ymax + 1, xmin:xmax + 1]
        mask_erode = mask_erode[ymin:ymax + 1, xmin:xmax + 1]
        cys_mask = cys_mask[ymin:ymax + 1, xmin:xmax + 1]

        if self.is_debug:
            cv2.imshow("dst", dst)

        orinodule_th, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thetas = [0, np.pi / 2]
        img_pym = self.img_pyramid(dst)
        dst = cv2.resize(img_pym, dst.shape[::-1])
        orient_img = self.gabor_reinforce(img_pym, 9, 1, thetas, np.pi)
        orient_img_ori = cv2.resize(orient_img, dst.shape[::-1])
        ridge_map = self.get_ridge_map(orient_img)
        ridge_map = cv2.resize(ridge_map, mask.shape[::-1], cv2.INTER_NEAREST)
        ridge_map = self.rm_small_cnts(ridge_map)

        ridge_thin = self.thinning(ridge_map)
        if self.is_debug:
            cv2.imshow("after reinforce", orient_img)
            cv2.imshow("ridge map", ridge_map)
            cv2.imshow("ridge map after thinning", ridge_thin)

        # 去除边界区域的ridge
        ridge_thin = cv2.bitwise_and(ridge_thin, mask_erode)
        if self.is_debug:
            cv2.imshow("ridge map after correct", ridge_thin)

        # 对细化后的ridge寻找垂直线
        ridge_thin_copy = copy.deepcopy(ridge_thin)

        # 用[-1, 2, -1]做全图卷积
        wait4del = []
        for row in range(1, ridge_thin_copy.shape[0] - 1):
            for col in range(1, ridge_thin_copy.shape[1] - 1):
                if ridge_thin_copy[row, col] == 255 and ridge_thin_copy[row, col - 1] == 0 and \
                        ridge_thin_copy[row, col + 1] == 0:
                    pass
                else:
                    wait4del.append((row, col))
        for row, col in wait4del:
            ridge_thin_copy[row, col] = 0

        if self.is_debug:
            cv2.imshow("ridge vertical", ridge_thin_copy)

        # 每个连通域的col跨度不能太大, 找横线
        # ridge_vertical_cnts, _ = cv2.findContours(ridge_thin_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #
        # if len(ridge_vertical_cnts) == 0:
        #     return 0
        rows, cols = ridge_thin_copy.shape
        comet_cnts_ver = []
        # for cnt in ridge_vertical_cnts:
        #     xmin = min(cnt[:, 0, 0])
        #     ymin = min(cnt[:, 0, 1])
        #     xmax = max(cnt[:, 0, 0])
        #     ymax = max(cnt[:, 0, 1])
        #     w = xmax - xmin
        #     h = ymax - ymin
        #     if w > 3 or h < 5:
        #         for x, y in cnt[:, 0, :]:
        #             ridge_thin_copy[y, x] = 0
        #             continue
        # if self.is_debug:
        #     cv2.imshow("after correct vertical", ridge_thin_copy)

        *_, ridge_vertical_cnts, _ = cv2.findContours(ridge_thin_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(ridge_vertical_cnts) == 0:
            return 0

        for cnt in ridge_vertical_cnts:
            # ymin = min(cnt[:, 0, 1])
            # ymax = max(cnt[:, 0, 1])
            # ver_len = ymax - ymin
            # x_p = 0  # 垂线最上方
            # x_p2 = 0  # 垂线的最下方
            # flag = [0, 0]
            # for point in cnt[:, 0, :]:
            #     if point[1] == ymin:
            #         x_p = point[0]
            #         flag[0] = 1
            #     if point[1] == ymax:
            #         x_p2 = point[0]
            #         flag[1] = 1
            #     if flag[0] == 1 and flag[1] == 1:
            #         break
            # if x_p == 0 or x_p2 == 0:
            #     print("Error: not found max y or min y")
            #     return 0
            # y_p = ymin
            # y_p2 = ymax
            line_points = {}
            for point in cnt[:, 0, :]:
                line_points[point[1]] = point[0]
            line_points_sorted_by_y = sorted(line_points.items(), key=operator.itemgetter(0))
            line_points_sorted_by_x = sorted(line_points.items(), key=operator.itemgetter(1))
            # print(line_points_sorted)
            y_p, x_p = line_points_sorted_by_y[0]
            y_p2, x_p2 = line_points_sorted_by_y[-1]
            _, xmin = line_points_sorted_by_x[0]
            _, xmax = line_points_sorted_by_x[-1]
            ver_len = y_p2 - y_p
            h_len = xmax - xmin
            if ver_len < 5 or h_len > 3:
                continue

            # 当垂线的顶端，即候选的T型交点，灰度值低于200或垂线底部灰度高于顶部时，排除这根垂线
            if orient_img_ori[y_p, x_p] < 230 or dst[y_p2, x_p2] > dst[y_p, x_p]:
                continue


            y_mid = int((y_p + y_p2) / 2)
            intensity_up = []
            intensity_down = []
            for y in range(y_p, y_mid):
                intensity_up.append(dst[y, line_points[y]])
            for y in range(y_mid, y_p2+1):
                intensity_down.append(dst[y, line_points[y]])
            # print(intensity_up)
            # print(intensity_down)
            # exit()

            up_median = np.median(intensity_up)
            down_median = np.median(intensity_down)
            if up_median <= down_median:
                continue

            patience_count = 0  # 可以容忍横线和竖线相隔1个像素
            while 0 < x_p < cols - 1 and 0 < y_p < rows - 1 and patience_count < 2:
                # 当正上方有白点
                line_point = self.three_neigh_white(ridge_thin, x_p, y_p, "up")
                if line_point != -1:

                    # 当上方有白点时，则需要能往左和往右各延伸两个像素点，才算是一条横线
                    count_left = 0
                    point_stack = [line_point]  # 用栈来递归
                    while len(point_stack) != 0:
                        cur_point = point_stack.pop()
                        new_point = self.three_neigh_white(ridge_thin, cur_point[0], cur_point[1], "left")
                        if new_point == -1:
                            break
                        else:
                            count_left += 1
                            point_stack.append(new_point)

                    if count_left < 2:
                        break

                    count_right = 0
                    point_stack = [line_point]
                    while len(point_stack) != 0:
                        cur_point = point_stack.pop()
                        new_point = self.three_neigh_white(ridge_thin, cur_point[0], cur_point[1], "right")
                        if new_point == -1:
                            break
                        else:
                            count_right += 1
                            point_stack.append(new_point)
                    if count_right < 2:
                        break
                    line_len = count_right + count_left + 1  # 横线的长度， 左右的像素加中点

                    # 横线与竖线不能超过结节大小的0.2和0.3， 且横线不能超过竖线的4倍（主要是竖线在采取的过程中可能被截断，所以放宽一些）
                    if line_len / ver_len > 4 or line_len > cols * 0.2 or ver_len > rows * 0.3:
                        break

                    # 以上条件都满足，则这个cnt是彗星尾的垂线
                    comet_cnts_ver.append(cnt)
                    break
                else:
                    y_p -= 1
                    patience_count += 1
                    continue
        if len(comet_cnts_ver) == 0:
            return 0
        else:
            *_, cys_cnts, _ = cv2.findContours(cys_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(cys_cnts) == 0:
                comet_cnts_ver = []
            min_x = max_x = 0
            min_y = max_y = 0
            for cys_cnt in cys_cnts:
                x, y, w, h = cv2.boundingRect(cys_cnt)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
            cys_flag = 0
            del_index = []
            for index in range(len(comet_cnts_ver)):
                if min_x < comet_cnts_ver[index][0, 0, 0] < max_x and min_y < comet_cnts_ver[index][0, 0, 1] < max_y:
                    cys_flag = 1
                    break
                if cys_flag == 1:
                    break
                else:
                    del_index.append(index)
            for index in del_index:
                comet_cnts_ver.pop(index)
            if cys_flag == 0:
                comet_cnts_ver = []

        if mode == "find":
            result = np.zeros(ridge_thin.shape, np.uint8)
            *_, ridge_thin_cnts, _ = cv2.findContours(ridge_thin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            comet_cnts = []
            if len(ridge_thin_cnts) == 0:
                return 0
            for cnt in ridge_thin_cnts:
                for comet_cnt_ver in comet_cnts_ver:
                    if comet_cnt_ver[:, 0, :] in cnt:
                        comet_cnts.append(cnt)
            result = cv2.drawContours(result, comet_cnts_ver, contourIdx=-1, color=255, thickness=-1)
            if self.is_debug:
                cv2.imshow("comet", result)
            return np.asarray(result)
        elif mode == "report":
            return len(comet_cnts_ver)

    @staticmethod
    def three_neigh_white(image, x, y, turn):
        """ 根据给定的方向查看当前坐标下，某方向的三领域内是否有255，如果有，则返回这个坐标值，没有则返回-1"""
        if turn == "left":
            if image[y, x - 1] == 255:
                return x - 1, y
            elif image[y - 1, x - 1] == 255:
                return x - 1, y - 1
            elif image[y + 1, x - 1] == 255:
                return x - 1, y + 1
            return -1
        elif turn == "right":
            if image[y, x + 1] == 255:
                return x + 1, y
            elif image[y - 1, x + 1] == 255:
                return x + 1, y - 1
            elif image[y + 1, x + 1] == 255:
                return x + 1, y + 1
            return -1
        elif turn == "up":
            if image[y - 1, x] == 255:
                return x, y - 1
            elif image[y - 1, x - 1] == 255:
                return x - 1, y - 1
            elif image[y - 1, x + 1] == 255:
                return x + 1, y - 1
            return -1



##############################################################################
# 囊实性
##############################################################################
    def find_or_report_constitute(self, mode):
        """
        本方法有两种模式，”find“模式下会返回找到的囊性区域mask（与输入尺寸相同）； ”report“模式下会输出结节的构成。
        :param mode: "find" or "report"
        :return: if mode == "find":  mask: numpy array
                                     只返回囊性区域的mask，shape = img.shape
                 if mode == "report": consititute_flag:int
                                       返回结节的构成指标  0： 实性    1： 实性为主  2： 囊性为主  3： 囊性
        """
        img = copy.deepcopy(self.img)
        mask = copy.deepcopy(self.mask_erode)
        box = copy.deepcopy(self.box)
        dst = self.set_zero_mask(img, mask, 20)

        # 将所有输入都裁剪成box的区域
        xmin, ymin, xmax, ymax = self.find_box_point(box)
        dst = dst[ymin:ymax + 1, xmin:xmax + 1]
        mask = mask[ymin:ymax + 1, xmin:xmax + 1]

        self.shadows = self.read_and_find_shadows()
        shadows_mask = self.shadows[ymin:ymax + 1, xmin:xmax + 1]
        # dst = cv2.medianBlur(dst, 5)

        if self.is_debug:
            cv2.imshow("dst", dst)
            cv2.imshow("mask", mask)

        # 找到囊性区域的可疑区域
        ret, th = self.otsu_and_anti(dst, mask, 20)

        if self.is_debug:
            cv2.imshow("otsu_anti", th)

        # 计算图像梯度， 并中值滤波
        final = self.compute_grad(dst)
        final = self.midFilter(final, 9)
        # plt.imshow(final, cmap="gray")
        # plt.show()

        # 统计囊性可疑区域内的梯度幅值直方图
        # hist = np.zeros((int(np.max(final)) + 1, 1))
        # for i in range(final.shape[0]):
        #     for j in range(final.shape[1]):
        #         # 当此点同时在囊性可疑区域和声影区域时，则将此点剔除出囊性可疑区域
        #         if int(th[i, j]) == 255 and int(shadows_mask[i, j]) == 255:
        #             th[i, j] = 0
        th_sup = cv2.bitwise_and(th, shadows_mask)
        th[th_sup == 255] = 0
                # if th[i, j] == 255:
                #     hist[int(final[i, j]), 0] += 1
        if self.is_debug:
            cv2.imshow("substract shadow", th)

        # 得到梯度为0区域的图
        liantong = np.zeros((final.shape[0], final.shape[1]), np.uint8)
        liantong[final == 0] = 255
        liantong[th != 255] = 0

        # 做开运算，使面积小的连通区域消失
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # liantong = cv2.erode(liantong, kernel)
        # liantong = cv2.morphologyEx(liantong, cv2.MORPH_OPEN, kernel)
        if self.is_debug:
            cv2.imshow("liantong", liantong)

        # 找到所有的连通域，并找到面积最大的那个连通域
        liantong_copy = copy.deepcopy(liantong)
        *_, contours, _ = cv2.findContours(liantong_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 区域生长法
        # if contours:
        #     seeds = []
        #     for cnt in contours:
        #         seeds.append(Point(cnt[int(cnt.shape[0] / 2), 0, 0], cnt[int(cnt.shape[0] / 2), 0, 1]))
        #     import time
        #     start = time.time()
        #     seedimg = self.regionGrow(dst, seeds, 0, mask)
        #     print(time.time()-start)
        #     cv2.imshow("1", seedimg)
        #     cv2.waitKey(0)
        # else:
        #     seedimg = liantong
        seedimg = liantong
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seedimg = cv2.erode(seedimg, kernel)
        seedimg[shadows_mask == 255] = 0
        if self.is_debug:
            cv2.imshow("seed", seedimg)
        # 找到连通域
        seedimg_copy = copy.deepcopy(seedimg)
        *_, contours, _ = cv2.findContours(seedimg_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # 求mask的区域点
        result = np.zeros(img.shape, np.uint8)
        cystica_cnts = []
        mask_erode_copy = copy.deepcopy(self.mask)
        *_, cnts, _ = cv2.findContours(mask_erode_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        sub_cnt_lens = [len(cnt) for cnt in cnts]
        max_idx = np.argmax(sub_cnt_lens)
        area_mask = cv2.contourArea(cnts[max_idx])
        cystica_area = 0

        # 当区域生长后的连通域大小超过mask区域面积的0.001时，把此连通域判定为囊性区域，因为找到的囊性区域往往比实际囊性区域要小一点
        if contours:
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area > 0.001 * area_mask:
                    cystica_cnts.append(contour)
                    cystica_area += contour_area
        # print(area_mask)
        # print(cystica_area)

        # 将囊性区域坐标还原到函数输入的图像尺寸
        for cnt in cystica_cnts:
            for pt in cnt:
                pt[:, 0] += xmin
                pt[:, 1] += ymin
        result = cv2.drawContours(result, cystica_cnts, contourIdx=-1, color=255, thickness=-1)
        if mode == "find":
            return np.asarray(result)

        if mode == "report":
            flag = self._fuzzy_constitute()
            if flag == 1:
                if cystica_area > area_mask * 0.4:
                    return 2, result

                elif cystica_area > area_mask * 0.01 or cystica_area > 100:
                    return 1, result

                # 实性
                else:
                    return 0, result

            else:
                if cystica_area < area_mask * 0.05:
                    return 0, result

                # 囊性为主
                if cystica_area < area_mask * 0.98:
                    return 2, result

                # 囊性
                else:
                    return 3, result

    def _fuzzy_constitute(self):
        """
        模糊地将结节分为偏囊性和偏实性
        :return: 1 偏实性  2 偏囊性
        """
        img = copy.deepcopy(self.img)
        mask = copy.deepcopy(self.mask_erode)
        img = cv2.medianBlur(img, 3)
        if self.is_debug:
            cv2.imshow("after_Median", img)

        if self.is_debug:
            cv2.imshow("after_erode", self.mask_erode)

        # 将mask铺在img上，并对结节区域做大津法分割
        dst = cv2.bitwise_and(img, mask)

        ret, th = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.bitwise_or(th, self.shadows)
        th = cv2.bitwise_and(th, self.mask_erode)

        if self.is_debug:
            cv2.imshow("after_OTSU", th)

        # 计算囊性和实性区域面积
        hist = cv2.calcHist([th], [0], self.mask_erode, [256], [0, 256])
        cystica_area = hist[0][0]
        solid_area = hist[-1][0]

        if solid_area > cystica_area:
            return 1
        else:
            return 2


###################################################################################
# 声影检测
###################################################################################
    def read_and_find_shadows(self):
        """
        提取声影
        :return: 声影的mask，声影区域为白色，其他部分为黑色
        """
        img = copy.deepcopy(self.img)
        mask = copy.deepcopy(self.mask_erode)
        box = copy.deepcopy(self.box)
        # 提取金标准box的四个点，主要是防止landmark对分割的影响
        xmin, ymin, xmax, ymax = self.find_box_point(box)

        # 将原图中mask轮廓以外的区域涂黑， 存为dst
        dst = self.set_zero_mask(img, mask, 10)

        if self.is_debug:
            cv2.imshow("after_crop", img)
            cv2.imshow("dst", dst)

        # 对dst这张图做大津分割，也就是说分割时不会受到mask区域以外的影响
        otsu_th, otsu_dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.is_debug:
            cv2.imshow("otsu_dst", otsu_dst)

        # 将上一步得到的阈值，对原图做二值分割，这样的话，就可以将黑色部分大致看成囊性区域或者是声影区域
        img_sub = img[ymin:, xmin:xmax]
        _, otsu_sub = cv2.threshold(img_sub, otsu_th, 255, cv2.THRESH_BINARY)
        otsu_sub = 255 - otsu_sub
        otsu_not = np.zeros(dst.shape, np.uint8)
        otsu_not[ymin:, xmin:xmax] = otsu_sub
        otsu = 255 - otsu_not

        if self.is_debug:
            cv2.imshow("otsu", otsu)

        # 将二值图反相，方便寻找原来为黑色部分的连通域
        # otsu_not = copy.deepcopy(otsu)
        # cv2.bitwise_not(otsu, otsu_not)
        otsu_not = self._erode(otsu_not, 3)
        if self.is_debug:
            cv2.imshow("otsu_not", otsu_not)

        # 求mask的外接矩形
        x, y, w, h = self.box_point

        # 对after open二值图从上往下遍历，去除囊性可疑区域
        for col in range(x, x + w):
            start = -1
            cur = -1
            count = 0
            for row in range(y, otsu_not.shape[0]):
                # start为-1且出现声影可疑区域在mask范围内时，记为声影开始,否则这列无声影
                if start == -1:
                    if row < (y + h):
                        if otsu_not[row, col] == 255:
                            start = row
                            cur = row
                    else:
                        start = y + h
                        break
                # 当声影已经开始且当前坐标仍然为声影可疑区域
                if start != -1 and otsu_not[row, col] == 255:
                    # 当遍历行数未超过mask往下50个像素点时，声影的当前指针继续往下
                    if cur < (y + h + 50):
                        count = 0
                        cur = row
                    # 当超过后，则认为由start往下都是声影
                    else:
                        break
                # 当声影已经开始，且当前遍历到的点不是声影可疑区域
                if start != -1 and otsu_not[row, col] == 0:
                    if count > 10:
                        start = -1
                    else:
                        count += 1
                        cur = row
            if start == -1:
                continue
            for row in range(y, start):
                otsu_not[row, col] = 0

        if self.is_debug:
            cv2.imshow("after linear probe", otsu_not)

        # 对可疑声影区域做一下膨胀，因为在上一步时在声影内部造成了许多细微小点
        otsu_open = self._dilate(otsu_not, 3)

        # 把声影区域限定在box内
        # for row in range(otsu_open.shape[0]):
        #     for col in range(otsu_open.shape[1]):
        #         if row > ymin and xmin < col < xmax:
        #             pass
        #         else:
        #             otsu_open[row, col] = 0
        otsu_open[:ymin, :] = 0
        otsu_open[:, :xmin] = 0
        otsu_open[:, xmax:] = 0

        if self.is_debug:
            cv2.imshow("after box restrict", otsu_open)

        # 寻找连通域
        otsu_open_copy = copy.deepcopy(otsu_open)
        *_, contours, _ = cv2.findContours(otsu_open_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        shadow_cnts = []

        # 如果一个连通域部分在mask区域内，部分在mask区域外，则认为这个连通域是一个声影，记下来
        if contours:
            for cnt in contours:
                shadow_flag = [0, 0]  # [0]代表有不在mask内的点， [1]代表有在mask内的点， 两者同时满足时， 这个cnt才是穿过mask的
                for pt in cnt:
                    flag = cv2.pointPolygonTest(self.cnt, tuple(pt[0]), False)
                    if flag == -1:
                        shadow_flag[0] = 1
                    else:
                        shadow_flag[1] = 1
                    if shadow_flag[0] == 1 and shadow_flag[1] == 1:
                        shadow_cnts.append(cnt)
                        break

        if self.is_debug:
            shadow_display = np.zeros(otsu.shape, np.uint8)
            shadow_display = cv2.drawContours(shadow_display, shadow_cnts, contourIdx=-1, color=255, thickness=-1)
            cv2.imshow("shadows_area", shadow_display)

        if self.is_debug:
            # 将声影部分在二值图中涂为白色
            after_otsu = cv2.drawContours(otsu, shadow_cnts, contourIdx=-1, color=255, thickness=-1)
            cv2.imshow("after", after_otsu)

        # 将声影坐标恢复成原图尺寸
        # for cnt in shadow_cnts:
        #     for point in cnt:
        #         point[:, 0] += min_col
        #         point[:, 1] += min_row

        result = np.zeros(img.shape, np.uint8)
        result = cv2.drawContours(result, shadow_cnts, contourIdx=-1, color=255, thickness=-1)
        if self.is_debug:
            cv2.imshow("origin after", result)
        return result

################################################################################
# 金字塔分解、ridge detection
################################################################################
    @staticmethod
    def img_pyramid(img, is_debug=False):
        """构建图像金字塔，返回平滑后的图像,返回的尺寸是原图1/2"""
        image = copy.deepcopy(img)
        G0 = image
        # G0_shape = G0.shape
        G1 = cv2.GaussianBlur(G0, (3, 3), 1)
        G1 = G1[::2, ::2]
        G1_shape = G1.shape
        G2 = cv2.GaussianBlur(G1, (3, 3), 1)
        G2 = G2[::2, ::2]
        # G2_shape = G2.shape

        L2 = cv2.resize(G2, G1_shape[::-1], interpolation=cv2.INTER_NEAREST)
        L2 = cv2.GaussianBlur(L2, (3, 3), 1)
        L2 = G1 - L2
        # L1 = cv2.resize(L2, G0_shape[::-1], interpolation=cv2.INTER_LINEAR)
        # L1 = cv2.GaussianBlur(L1, (3, 3), 1)
        # L1 = G0 - L1

        LP2 = cv2.resize(G2, G1_shape[::-1], interpolation=cv2.INTER_NEAREST)
        LP2 = cv2.GaussianBlur(LP2, (3, 3), 1)
        LP2 = LP2 + L2
        # LP1 = cv2.resize(LP2, G0_shape[::-1], interpolation=cv2.INTER_NEAREST)
        # LP1 = cv2.GaussianBlur(LP1, (3, 3), 1)
        # cv2.waitKey(0)
        return LP2

    @staticmethod
    def gabor_reinforce(img, ksize, sigma, theta_list, wavelen):
        """ 小波滤波，方向增强"""
        kernels = []
        for theta in theta_list:
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, wavelen, 0.5, 0, ktype=cv2.CV_32F)
            # kern /= kern.sum()
            kernels.append(kern)

        gabor_imgs = [
            cv2.filter2D(
                img,
                ddepth=-1,
                kernel=gabor_k) for gabor_k in kernels]

        for i in range(len(gabor_imgs)):
            gabor_imgs[i] = cv2.GaussianBlur(gabor_imgs[i], (3, 3), 0)
            gabor_imgs[i] = cv2.medianBlur(gabor_imgs[i], 3)

        orient_img = gabor_imgs[0]
        for gabor_img in gabor_imgs[1:]:
            orient_img = np.maximum(orient_img, gabor_img)

        return orient_img


    @staticmethod
    def get_hessian_matrix(image):
        """
        求图像的Hessian矩阵，注意此方法里没有高斯平滑，所以请在调用之前先做好平滑操作
        求导方式使用一阶中心差分
        H = [ Hrr Hrc Hcc]
        """
        img_float = image.astype(float)
        gradients = np.gradient(img_float)  # 返回两个梯度图，分别是col方向和row方向
        Ix, Iy = gradients
        axes = range(img_float.ndim)  # 图的维数
        # np.grandient是按照x、y方向的顺序保存梯度图的， 与row、col方向刚好相反，所以要取反
        axes = reversed(axes)
        H = [np.gradient(gradients[ax0], axis=ax1)
             for ax0, ax1 in combinations_with_replacement(axes, 2)]
        return Ix, Iy, H

    def get_hessian_eigvals(self, H_elems):
        """
        计算Hessian矩阵的特征值,返回特征值最大的图像矩阵和特征值最小的图像矩阵
        :param H_elems:
        :return: max 、 min
        """
        matrices = self._get_hessian_matrix_image(H_elems)
        # eigvalsh returns eigenvalues in increasing order. We want decreasing
        # eigvals = np.linalg.eigvalsh(matrices)[..., ::-1]
        eigvals, eigvectors = np.linalg.eigh(matrices)
        eigvals = eigvals[..., ::-1]
        print(eigvals.shape)
        print(eigvectors)
        leading_axes = tuple(range(eigvals.ndim - 1))
        eigvals = np.transpose(eigvals, (eigvals.ndim - 1,) + leading_axes)

        return eigvals

    @staticmethod
    def _get_hessian_matrix_image(H_elems):
        """
        将get_hessian_matirx中得到的海塞矩阵中的每个方向的子矩阵，合并成一个高维矩阵
        维度为 (img.shape[0], img.shape[1], img.ndim, img.ndim)
        """
        image = H_elems[0]
        hessian_image = np.zeros(image.shape + (image.ndim, image.ndim))
        for idx, (row, col) in \
                enumerate(combinations_with_replacement(range(image.ndim), 2)):
            hessian_image[..., row, col] = H_elems[idx]
            hessian_image[..., col, row] = H_elems[idx]
        return hessian_image

    def get_ridge_map(self, img):
        """对图像做ridge detection， 并返回ridge图"""
        m, n = img.shape
        Ix, Iy, H_elems = self.get_hessian_matrix(img)
        # img_float = orient_img.astype(float)
        # Ix, Iy = np.gradient(img_float)
        # Ixx, Ixy = np.gradient(Ix)
        # Iyx, Iyy = np.gradient(Iy)
        vec = np.zeros((m, n, 2))
        val = np.zeros((m, n))
        Q = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                H = np.empty((2, 2))
                H[0, 0] = H_elems[2][i, j]
                H[0, 1] = H_elems[1][i, j]
                H[1, 0] = H_elems[1][i, j]
                H[1, 1] = H_elems[0][i, j]
                d, v = np.linalg.eig(H)
                if abs(d[0]) > abs(d[1]):
                    vec[i, j, 0] = v[0, 0]
                    vec[i, j, 1] = v[1, 0]
                    val[i, j] = d[0]
                else:
                    vec[i, j, 0] = v[0, 1]
                    vec[i, j, 1] = v[1, 1]
                    val[i, j] = d[1]
                Q[i, j] = vec[i, j, 0] * Ix[i, j] + vec[i, j, 1] * Iy[i, j]

        ridge_map = np.ones((m, n), np.uint8) * 255

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if val[i, j] < 0 and abs(val[i, j]) > 10:
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if k != 0 and l != 0:
                                Qy = Q[i + k, j + l]
                                P1 = vec[i, j, 0] * vec[i + k, j + l, 0] + \
                                     vec[i, j, 1] * vec[i + k, j + l, 1]
                                P1 = Q[i, j] * Qy * P1
                                if P1 < 0:
                                    P2 = Q[i, j] * (vec[i, j, 0] * Ix[i + k, j + l] + vec[i, j, 1] * Iy[i + k, j + l])
                                    if P2 < 0:
                                        ridge_map[i, j] = 0
        ridge_map = 255 - ridge_map
        return ridge_map

    @staticmethod
    def rm_small_cnts(img):
        """ 删去ridge中一些较小的噪声线"""
        img_copy = copy.deepcopy(img)
        *_, cnts, _ = cv2.findContours(
            img_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img_return = np.zeros(img.shape, np.uint8)
        ridge_cnts = []
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 and h < 10:
                continue
            else:
                ridge_cnts.append(cnt)
        img_return = cv2.drawContours(
            img_return,
            ridge_cnts,
            contourIdx=-1,
            color=255,
            thickness=-1)
        return img_return

################################################################################
# 骨架提取
################################################################################

    @staticmethod
    def neighbours(img, r, c):
        """
        获取当前像素点的八邻域点的像素值
        :param img: 图
        :param r: 当前点的行
        :param c: 当前点的列
        :return: list
        """
        return [img[r-1, c], img[r-1, c+1], img[r, c+1], img[r+1, c+1],
                img[r+1, c], img[r+1, c-1], img[r, c-1], img[r-1, c-1]]

    @staticmethod
    def sum_zero2one(neighbours):
        """返回邻域中像素点按0-1顺时针变化的次数"""
        n = neighbours + neighbours[0:1]
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

    def thinning(self, image):
        """细化二值图的连通域"""
        img = copy.deepcopy(image)
        img_sub = copy.deepcopy(img)
        img[img != 0] = 1
        flag1 = flag2 = 1  # 每次循环有两次迭代，所以设置两个待删除标识
        while flag1 or flag2:
            # 第一次迭代
            flag1 = []
            img_sub[:, :] = 1
            rows, cols = img.shape
            # 长宽都留一个像素边界，因为没有做padding
            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    if img[row, col] != 1:
                        continue
                    p2, p3, p4, p5, p6, p7, p8, p9 = n = self.neighbours(img, row, col)
                    if 2 <= sum(n) <= 6 and self.sum_zero2one(n) == 1 and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                        # flag1.append((row, col))
                        flag1 = 1
                        img_sub[row, col] = 0
            # for (row, col) in flag1:
            #     img[row, col] = 0
            img = cv2.bitwise_and(img, img_sub)

            # 第二次迭代
            flag2 = []
            img_sub[:, :] = 1
            rows, cols = img.shape
            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    if img[row, col] != 1:
                        continue
                    p2, p3, p4, p5, p6, p7, p8, p9 = n = self.neighbours(img, row, col)
                    if 2 <= sum(n) <= 6 and self.sum_zero2one(n) == 1 and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                        # flag2.append((row, col))
                        flag2 = 1
                        img_sub[row, col] = 0
            # for (row, col) in flag2:
            #     img[row, col] = 0
            img = cv2.bitwise_and(img, img_sub)

        img[img != 0] = 255
        return img

########################################################################################
####回声水平
########################################################################################

    def compute_nodule_echo(self, mask_all_nodules=None, mask_calcification=None, mask_cyst=None):
        '''
        超声图像回声水平判断
        :param mask_all_nodules: 所有结节mask图像（已脱敏），numpy数组，dtype=np.uint8，图中列出图像中所有结节（多发情况），默认为None
        :param mask_calcification: 结节钙化部分mask图像（已脱敏），numpy数组，dtype=np.uint8，默认为None
        :param mask_cyst: 结节囊性部分mask图像（已脱敏），numpy数组，dtype=np.uint8，默认为None
        :return: 返回int型检测结果，其数值代表的回升水平可由ClassifyEchoType.return_num_to_string获得，返回-1表示回声判断异常
        '''
        image = copy.deepcopy(self.img)
        mask_target_nodule = copy.deepcopy(self.mask)
        mask_acoustic_shadow = copy.deepcopy(self.shadows)
        mask_calcification = np.zeros(mask_cyst.shape, np.uint8)

        image = cv2.medianBlur(image, 3)
        shape_image = image.shape

        mask_muscular = self.find_muscular(image)

        if len(shape_image) != 2:
            print('Error: wrong image shape !')
            return -1

        if self.is_debug:
            cv2.imshow('image', image)
            cv2.imshow('mask_target_nodule', mask_target_nodule)
            if mask_all_nodules is not None:
                cv2.imshow('mask_all_nodules', mask_all_nodules)
            if mask_acoustic_shadow is not None:
                cv2.imshow('mask_acoustic_shadow', mask_acoustic_shadow)
            if mask_calcification is not None:
                cv2.imshow('mask_calcification', mask_calcification)
            if mask_cyst is not None:
                cv2.imshow('mask_cyst', mask_cyst)

        mask_target_nodule_copy = copy.deepcopy(mask_target_nodule)
        *_, contours, _ = cv2.findContours(mask_target_nodule_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            print('Error: contours not found in mask_target_nodule.')
            return -1
        elif len(contours) > 1:
            area = []
            for cnt in contours:
                area.append(cv2.contourArea(cnt))
            # print('Error: multiple contours found in mask_target_nodule.')
            # return -1
            tmp_mask = np.zeros(shape_image, dtype=np.uint8)
            cv2.drawContours(tmp_mask, contours, area.index(max(area)), 255, thickness=-1)
            mask_target_nodule = tmp_mask
        if mask_all_nodules == None:
            mask_all_nodules = copy.deepcopy(mask_target_nodule)
        mask_union = np.zeros(shape_image, dtype=np.uint8)
        mask_union[(mask_target_nodule > 0) | (mask_all_nodules > 0) | (mask_acoustic_shadow > 0) |
                   (mask_calcification > 0) | (mask_cyst > 0) | (mask_muscular > 0)] = 255
        # 目标结节mask图像，所有结节mask图像，声影mask图像，结节钙化部分mask图像，结节囊性部分mask图像，目标是找到甲状腺非结节区域
        mask_union_exclude_target_nodule = np.zeros(shape_image, dtype=np.uint8)
        mask_all_nodules_exclude_target = copy.deepcopy(mask_all_nodules)
        mask_all_nodules_exclude_target[mask_target_nodule > 0] = 0
        mask_union_exclude_target_nodule[(mask_all_nodules_exclude_target > 0) | (mask_acoustic_shadow > 0) |
                                         (mask_calcification > 0) | (mask_cyst > 0)] = 255
        # 非所有结节mask图像，声影mask图像，结节钙化部分mask图像，结节囊性部分mask图像，目的是找到目标结节实性区域
        if self.is_debug:
            cv2.imshow('mask_union', mask_union)

        mean_intensity_thyroid = self._compute_thyroid_intensity(image, mask_target_nodule, mask_union)

        if mean_intensity_thyroid == -1:
            if self.is_debug:
                cv2.waitKey(0)
            return -1

        mean_intensity_nodule = self._compute_nodule_intensity(image, mask_target_nodule,
                                                                           mask_union_exclude_target_nodule)

        if mean_intensity_nodule == -1:
            if self.is_debug:
                cv2.waitKey(0)
            return -1
        # 调参标记
        A = 1.25
        B = 0.95
        C = 0.15
        D = 10
        if mean_intensity_nodule > mean_intensity_thyroid * A:
            # print('高回声')
            return_code = 4
        elif mean_intensity_nodule < mean_intensity_thyroid * B:
            if mean_intensity_nodule < min(mean_intensity_thyroid * C, D):
                # print('无回声')
                return_code = 0
            else:
                # print('低回声')
                return_code = 2
        else:
            # print('等回声')
            return_code = 3

        if self.is_debug:
            cv2.waitKey(0)

        return return_code

    def _compute_thyroid_intensity(self, image, mask_target_nodule, mask_union):
        image = copy.deepcopy(image)
        # 脱敏图像可能比ROI(超声图像中，真实图像区域，去除周围敏感信息)大，先对图像适当裁剪
        tmp_param = 0.1
        (len_row, len_col) = shape_image = image.shape
        row_left = int(len_row * tmp_param)
        row_right = int(len_row * (1 - tmp_param))
        col_left = int(len_col * tmp_param)
        col_right = int(len_col * (1 - tmp_param))
        tmp_mask = image[row_left:row_right, col_left:col_right]
        # 图像中最暗与最亮的部分都不是腺体
        value_min = tmp_mask.min() + 10
        value_max = int(tmp_mask.max() * 0.8)

        tmp_mask = np.zeros((len_row, len_col), dtype=np.uint8)
        # tmp_img中，灰度值在 value_min 与 value_max 之间的像素为黑色，否则为白色
        tmp_mask[(image < value_min) | (image > value_max)] = 255
        kernel = np.ones((3, 3), np.uint8)
        # 膨胀后腐蚀
        tmp_mask = cv2.morphologyEx(tmp_mask, cv2.MORPH_CLOSE, kernel)

        if self.is_debug:
            cv2.imshow('pixel value < value_min, > value_max', tmp_mask)

        mask_RONI = cv2.bitwise_or(tmp_mask, mask_union)
        # mask_ROI 可能是腺体的图像mask
        mask_ROI = cv2.bitwise_not(mask_RONI)

        if self.is_debug:
            cv2.imshow('mask_ROI', mask_ROI)

        # TODO: OTSU选择
        # 将剩余图像高低灰度区分，取二值化阈值
        # thres_OTSU = OTSU.get_threshold(image, mask_roi=mask_ROI)
        # mask_OTSU = np.zeros(shape_image, dtype=np.uint8)
        # mask_OTSU[(mask_ROI > 0) & (image >= thres_OTSU)] = 255
        dst = cv2.bitwise_and(image, mask_ROI)
        thres_OTSU, mask_OTSU = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        # mask_OTSU = cv2.morphologyEx(mask_OTSU, cv2.MORPH_OPEN, kernel)

        mask_OTSU_inverse =  np.zeros(shape_image, dtype=np.uint8)
        mask_OTSU_inverse[(mask_ROI > 0) & (image < thres_OTSU)] = 255

        if self.is_debug:
            cv2.imshow('mask_OTSU', mask_OTSU)
            cv2.imshow('mask_OTSU_inverse', mask_OTSU_inverse)

        area_nodule = np.count_nonzero(mask_target_nodule)
        equi_radius = int(np.sqrt(area_nodule / np.pi))
        size_kernel = max(int(equi_radius * 0.10), 1) * 2 + 1
        kernel = np.ones((size_kernel, size_kernel), np.uint8)
        # mask膨胀
        mask_dilate = cv2.dilate(mask_target_nodule, kernel)
        # mask膨胀再外扩一圈
        tmp_mask = cv2.dilate(mask_dilate, kernel)
        # tmp_mask 结节往外一圈，排除不合要求的像素
        tmp_mask[(mask_dilate > 0) | (mask_RONI > 0)] = 0

        if self.is_debug:
            cv2.imshow('mask_ring', tmp_mask)

        #S1面积计算
        count_ring = np.count_nonzero(tmp_mask)
        if count_ring == 0:
            print('Error: count_ring == 0')
            return -1

        # 找到ring中大于otsu阈值的像素数目
        tmp_mask[mask_OTSU == 0] = 0

        if self.is_debug:
            cv2.imshow('mask_ring_and_otsu', tmp_mask)
        #S2面积计算
        count_ring_white = np.count_nonzero(tmp_mask)

        if self.is_debug:
            print('count_ring: %d count_ring_white: %d count_ring_white/count_ring: %f' % (count_ring, count_ring_white, count_ring_white / count_ring))

        if count_ring_white / count_ring < 0.3:
            mask_thyroid = mask_OTSU_inverse
            if self.is_debug:
                print('mask_thyroid is mask_OTSU_inverse.')
        else:
            mask_thyroid = mask_OTSU
        size_kernel = 5
        kernel = np.ones((size_kernel, size_kernel), np.uint8)
        # mask_thyroid = cv2.morphologyEx(mask_thyroid, cv2.MORPH_OPEN, kernel)  # erosion, dilation
        mask_thyroid = cv2.morphologyEx(mask_thyroid, cv2.MORPH_CLOSE, kernel)  # dilation, erosion

        # compute thyroid mean intensity
        equi_radius = max(min(int(equi_radius * 0.2), 20), 10)
        kernel_size = equi_radius * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        tmp_mask = cv2.dilate(mask_dilate, kernel)
        tmp_mask[(mask_dilate > 0) | (mask_thyroid == 0)] = 0
        mask_thyroid = tmp_mask

        if self.is_debug:
            cv2.imshow('mask_thyroid', mask_thyroid)

        if np.count_nonzero(mask_thyroid) < 100:
            print('Error: compute thyroid intensity failure. Valid pixel number: %d.' % np.count_nonzero(mask_thyroid))
            return -1

        sorted_pixels = np.sort(image[mask_thyroid > 0])
        left = int(len(sorted_pixels) * 0.05)
        right = int(len(sorted_pixels) * 0.95)
        mean_intensity_thyroid = np.mean(sorted_pixels[left:right])
        # mean_intensity_thyroid_1 = np.mean(image[mask_thyroid > 0])

        if self.is_debug:
            print('mean_intensity_thyroid: %d' % mean_intensity_thyroid)

        return mean_intensity_thyroid

    def _compute_nodule_intensity(self, image, mask_target_nodule, mask_union_exclude_target_nodule):
        image = copy.deepcopy(image)
        area_nodule = np.count_nonzero(mask_target_nodule)
        equi_radius = np.sqrt(area_nodule / np.pi)
        size_kernel = int(max(equi_radius * 0.1, 1)) * 2 + 1
        kernel = np.ones((size_kernel, size_kernel), np.uint8)
        mask_nodule = cv2.erode(mask_target_nodule, kernel)
        mask_nodule[mask_union_exclude_target_nodule > 0] = 0

        if self.is_debug:
            cv2.imshow('mask_nodule', mask_nodule)

        if np.count_nonzero(mask_nodule) == 0:
            print('Error: compute nodule intensity failure. Valid pixel number: %d.' % np.count_nonzero(mask_nodule))

            if self.is_debug:
                cv2.waitKey(0)

            return -1

        sorted_pixels = np.sort(image[mask_nodule > 0])
        left = int(len(sorted_pixels) * 0.05)
        right = int(len(sorted_pixels) * 0.95)
        mean_intensity_nodule = np.mean(sorted_pixels[left:right])
        # mean_intensity_nodule_1 = np.mean(image[mask_nodule > 0])

        if self.is_debug:
            print('mean_intensity_nodule: %d' % mean_intensity_nodule)

        return mean_intensity_nodule

    def find_muscular(self, img):
        """找肌肉层"""
        # 3000,10000,0.2
        mser = cv2.MSER_create(_min_area=3000, _max_area=10000, _max_variation=0.3)
        regions, boxes = mser.detectRegions(img)
        img1 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        img1 = cv2.fillPoly(img1, regions, 255)
        return img1

################################################################################
# 回声灶
################################################################################
    def find_or_report_calcification(self, pixels_one_cm):
        """
        查找钙化区域
        :param pixels_one_cm: 由标尺检测得出， 1厘米所占的像素数
        :param mode: "find" or "report"
        :return: if mode == "report": result (list) result[0]==1代表微钙化，result[1]==1代表粗钙化, result[2]==1代表环钙化
        """
        """
        查找钙化区域
        :param pixels_one_cm: 由标尺检测得出， 1厘米所占的像素数
        :param mode: "find" or "report"
        :return: if mode == "report": result (list) result[0]==1代表微钙化，result[1]==1代表粗钙化, result[2]==1代表环钙化
                 if mode == "find":  mask: numpy array
                                     只返回钙化区域的mask，shape = img.shape
        """
        img = self.img
        mask = self.mask
        cystica_result = self.find_or_report_constitute('find')
        shadows_result = self.read_and_find_shadows()
        if len(img.shape) != 2:
            print("Error: image shape is wrong")
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        img = cv2.medianBlur(img, 3)
        mask_copy = copy.deepcopy(mask)
        mask_contours, _ = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(mask_contours) == 0:
            print("-----the mask is None-----")
        elif len(mask_contours) > 1:
            max_index = -1
            max_area = -1
            for i, cnt in enumerate(mask_contours):
                cnt_area = cv2.contourArea(cnt)
                max_area = max(cnt_area, max_area)
                if max_area == cnt_area:
                    max_index = i
                mask = np.zeros(mask.shape, np.uint8)
                mask = cv2.drawContours(mask, [mask_contours[max_index]], contourIdx=-1, color=255, thickness=-1)
                
        ################################################################################
        # 取病灶区域
        ################################################################################       
        mask_copy = copy.deepcopy(mask)
        contours, _ = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            raise Exception("the input mask can not be all black")
        '''
        sub_cnt_lens = [len(cnt) for cnt in contours]
        max_idx = np.argmax(sub_cnt_lens)
        cnt = contours[max_idx]
        box = np.zeros(mask.shape, np.uint8)
        x, y, w, h = cv2.boundingRect(cnt)
        box_point = [x, y, w, h]
        box[y:y + h + 1, x:x + w + 1] = 255
        '''
        dst = cv2.bitwise_and(img, mask)
        dst = self.set_zero_mask(dst, mask, 10)
        #xmin, ymin, w, h = box_point
        
        xmin, ymin, w, h = self.box_point
        xmax = xmin + w
        ymax = ymin + h
        img_ = img[ymin:ymax + 1, xmin:xmax + 1]
        dst_ = dst[ymin:ymax + 1, xmin:xmax + 1]
        mask_ = mask[ymin:ymax + 1, xmin:xmax + 1]
        
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        
        mask_erode = cv2.erode(mask, kernel1)
        mask_dilate = cv2.dilate(mask, kernel2)
        dst = self.set_zero_mask(dst, mask_erode, 10)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        cystica_dilate = cv2.dilate(cystica_result, kernel)
        
        ##############################################################################
        # 剔除病灶中声影区域和囊性区域
        ##############################################################################
        nodule_dst = cv2.bitwise_and(img, mask_erode)
        nodule_dst_ = cv2.bitwise_and(img, mask_dilate)
        shadows_result_inv = cv2.bitwise_not(shadows_result)
        cystica_result_inv = cv2.bitwise_not(cystica_dilate)
        nodule_dst = cv2.bitwise_and(nodule_dst, shadows_result_inv)
        nodule_dst = cv2.bitwise_and(nodule_dst, cystica_result_inv)
        nodule_dst = nodule_dst[ymin:ymax + 1, xmin:xmax + 1]
        dst_mask = cv2.bitwise_and(mask, shadows_result_inv)
        dst_mask = dst_mask[ymin:ymax + 1, xmin:xmax + 1]
        
        ##############################################################################
        # superpixels
        ##############################################################################        
        dst_copy = np.array(dst_)
        dst_copy = Image.fromarray(dst_copy.astype('uint8'))

        dst_copy = np.array(dst_copy)
        labels = segmentation.slic(dst_copy, compactness=0.1, n_segments=dst_copy.shape[0]*dst_copy.shape[0]/20, min_size_factor = 0.7)
        g = graph.RAG(labels)
        adj = adj_matrix(g).todense()
        
        labels_nodule = segmentation.slic(img_, compactness=0.1, n_segments=10)
        g_nodule = graph.RAG(labels_nodule)
        adj_ = adj_matrix(g_nodule).todense()
        ################################################################################
        # 找出亮度前7%的像素点
        ################################################################################
        sum_gray = 0
        histtest  = np.zeros(256)
        for i in range(nodule_dst.shape[0]):
            for j in range(nodule_dst.shape[1]):
                if dst_mask[i,j] == 255:
                    histtest[nodule_dst[i,j]] += 1
                    sum_gray += 1

        sum_pixel = 0
        gray = 0 
        for k in range(256):
            sum_pixel += histtest[255-k]
            if sum_pixel >= sum_gray * 0.07:
                gray = 255-k
                break
        dst_copy = copy.deepcopy(nodule_dst)
        for i in range(dst_copy.shape[0]):
            for j in range(dst_copy.shape[1]):
                if dst_copy[i][j] < gray:
                    dst_copy[i][j] = 0
        
        dst_gray = np.zeros(nodule_dst.shape)
        index_need = []
        label_need = []
        gray_mean = []
        for p in range(adj.shape[0]):
            index = np.argwhere(labels == p)
            sumgray = 0
            sum_num = 0
            for i in range(len(index)):
                sumgray += nodule_dst[index[i][0]][index[i][1]]
                sum_num += 1
            gray_mean.append(sumgray/sum_num)
        for p in range(adj.shape[0]):
            neighbor = np.argwhere(adj[p] == 1)
            index = np.argwhere(labels == p)
            flag = 0
            for q in range(len(neighbor)):
                if gray_mean[p] < gray_mean[neighbor[q][1]]:
                    flag += 1
            if flag <= 2 and gray_mean[p] >= gray:
                label_need.append(p)
                for j in range(len(index)):
                    index_need.append(index[j])
        
        for i in range(len(index_need)):
            dst_gray[index_need[i][0]][index_need[i][1]] = nodule_dst[index_need[i][0]][index_need[i][1]]
        ################################################################################
        # 大superpixel直方图计算
        ################################################################################
        hist_large_pixel = []
        #sum_large_pixel = []
        m_large_pixel = []
        omiga_large_pixel = []
        for p in range(adj_.shape[0]):
            sum_ = 0    
            hist_  = np.zeros(256)
            cnt_back = np.argwhere(labels_nodule == p)
            for i in range(len(cnt_back)):
                if dst_gray[cnt_back[i][0]][cnt_back[i][1]] == 0 and img_[cnt_back[i][0]][cnt_back[i][1]] < 240: 
                    hist_[img_[cnt_back[i][0]][cnt_back[i][1]]] += 1
                    sum_ += 1
            #sum_large_pixel .append(sum_) 
            #大superpixel参数估计
            hist_ = hist_/float(sum_)
            #print(hist_)
            hist_large_pixel.append(hist_)
            omiga_ = 1e-5
            for i in range(256):
                omiga_ += i*i*hist_[i]
            #print(omiga_back)
            a_ = omiga_*omiga_
            b_ = 1e-5
            for i in range(256):
                b_ += ((i*i) - omiga_)*((i*i) - omiga_)*hist_[i]
            #print(a_back, b_back, a_back/b_back)
            m_ = a_/b_
            #y_back = 2*pow(m_back,m_back)*pow(x,2*m_back-1)*np.exp(-m_back/omiga_back*x*x)/(gamma(m_back)*pow(omiga_back,m_back))
            m_large_pixel.append(m_)
            omiga_large_pixel.append(omiga_)
            
        ################################################################################
        # 计算每个小superpixel属于那个大superpixel
        ################################################################################
        label_of_big_pixel = np.zeros(adj.shape[0])
        for p in range(adj.shape[0]):
            cnt = np.argwhere(labels == p)
            list_label = []
            for i in range(len(cnt)):
                list_label.append(labels_nodule[cnt[i][0]][cnt[i][1]])
            #print(list_label)
            set_label = set(list_label)
            set_label_ = list(set_label)
            #print(set_label)
            label_num = np.zeros(len(set_label))
            for q in range(len(set_label)):
                label_num[q] = list_label.count(set_label_[q])
            #print(set_label_[np.argmax(label_num)])
            label_of_big_pixel[p] = set_label_[np.argmax(label_num)]     
            
        ################################################################################
        # 比较选中区域和周围像素
        ################################################################################
        img_cal = np.zeros(nodule_dst.shape,np.uint8)
        cal_cnts = []
        sum_thy = 0
        gray_sum_thy = 0
        gray_mean_thy = 0
        hist_thy = np.zeros(256)
        for i in range(nodule_dst.shape[0]):
            for j in range(nodule_dst.shape[1]):
                if nodule_dst_[i,j] == 0:
                    hist_thy[img_[i,j]] += 1
                    sum_thy += 1
                    gray_sum_thy += img_[i,j]
        

        hist_thy = hist_thy/float(sum_thy)
        gray_mean_thy = gray_sum_thy/sum_thy
        x = np.arange(0,256,1)
        
        for l in range(len(label_need)):
            cnt =  np.argwhere(labels == label_need[l])
            sum_cal = 0
            gray_sum_cal = 0
            gray_mean_cal = 0
            hist_cal  = np.zeros(256)
            #hist_back  = np.zeros(256)
                            
            for i in range(len(cnt)):
                if mask_[cnt[i][0]][cnt[i][1]] != 0 and img_[cnt[i][0]][cnt[i][1]] < 240:
                    hist_cal[img_[cnt[i][0]][cnt[i][1]]] += 1
                    sum_cal += 1
                    gray_sum_cal += img_[cnt[i][0]][cnt[i][1]]
            
            label_num = int(label_of_big_pixel[label_need[l]])
            #hist_back = hist_large_pixel[label_num] 
            
            gray_mean_cal = gray_sum_cal/(sum_cal+1)
            hist_cal = hist_cal/float(sum_cal)
            x = np.arange(0,256,1) 
    
        ################################################################################
        # nakagami distribution
        ################################################################################
            m_back = m_large_pixel[label_num] 
            omiga_back = omiga_large_pixel[label_num] 
            #y_back = 2*pow(m_back,m_back)*pow(x,2*m_back-1)*np.exp(-m_back/omiga_back*x*x)/(gamma(m_back)*pow(omiga_back,m_back))
            
            omiga_cal = 1e-5
            for i in range(256):
                omiga_cal += i*i*hist_cal[i]
            #a_cal = omiga_cal*omiga_cal
            b_cal = 1e-5
            for i in range(256):
                b_cal += ((i*i) - omiga_cal)*((i*i) - omiga_cal)*hist_cal[i]
            #m_cal = a_cal/b_cal
            
            y_probability = 2*pow(m_back,m_back)*pow(gray_mean_cal,2*m_back-1)*np.exp(-m_back/omiga_back*gray_mean_cal*gray_mean_cal)/(gamma(m_back)*pow(omiga_back,m_back))
    
            if omiga_cal - omiga_back > 6000:
                if gray_mean_cal - gray_mean_thy > 40 and y_probability < 0.005:
                    cal_cnts.append(cnt[:,[1,0]])
            
        img_cal = cv2.drawContours(
                img_cal,
                cal_cnts,
                contourIdx=-1,
                color=255,
                thickness=-1)    
        '''
        cal_cnts = []
        cnts, _ = cv2.findContours(img_cal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < img_.shape[0]/20 or w < img_.shape[1]/20:
                if h < img_.shape[1]/20 or h < img_.shape[0]/20:
                    continue
            
            cal_cnts.append(cnt)
        '''    
        if mode == "find": 
            img_cal_ = np.zeros(nodule_dst.shape,np.uint8)
            for cnt in cal_cnts:
                for pt in cnt:
                    pt[:, 0] += xmin
                    pt[:, 1] += ymin
            img_cal_ = cv2.drawContours(img_cal_, cal_cnts, contourIdx=-1, color=255, thickness=-1)
            return np.asarray(img_cal_)

        if mode == "report":
            result = [0, 0, 0]  # result[0]代表微钙化， result[1]代表粗钙化，result[2]代表环钙化
            #cal_mask = img_cal
            contours, _ = cv2.findContours(img_cal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
            for contour in contours:
                rect = cv2.minAreaRect(contour)  # 得到外接最小旋转矩形
                length = max(rect[1])
    
                # 得到物理长度, 单位是厘米
                real_len = length / float(pixels_one_cm)
                #print(real_len)
                if real_len > 0.1:
                    result[1] = 1
                elif real_len <= 0.1:
                    result[0] = 1

            return result

################################################################################
# 边缘清晰、模糊
################################################################################
    def border_clear_fuzzy(self):
        """
        边界清晰、模糊
        :param self:
        :return: 1 - 清晰  0 - 模糊
        """
        return 1


################################################################################
# 其他工具
################################################################################

    @staticmethod
    def find_box_point(box):
        """
        输入一个box的mask，输出此box的左上角和右下角
        :param box:
        :return: xmin，ymin，xmax，ymax
        """
        *_, cnts, _ = cv2.findContours(box, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(cnts) == 0:
            raise Exception("\n\n ---------The jinbox mask cannot be all black-------\n\n")
        x, y, w, h = cv2.boundingRect(cnts[0])
        return x, y, x + w, y + h

    @staticmethod
    def set_zero_mask(img, mask, ret):
        """
        将img中对应mask为白色的部分，阈值小于ret的，设置其灰度为0。并且mask以外的区域也涂黑
        :param img:
        :param mask:
        :param ret:
        :return:
        """
        img = cv2.bitwise_and(img, mask)
        img[img < ret] = 0
        return img

    # 获取这张图的otsu的阈值和结果，并且结果是在mask区域进行灰度反相的结果
    @staticmethod
    def otsu_and_anti(dst, mask, thresh):
        dst = copy.deepcopy(dst)
        mask = copy.deepcopy(mask)
        ret, solid_mask = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # solid_mean, solid_stddev = cv2.meanStdDev(dst, mask=solid_mask)
        cystica_mask = copy.deepcopy(solid_mask)
        cystica_mask = 255 - cystica_mask
        cystica_mask[dst > thresh] = 0
        cystica_mask = cv2.bitwise_and(cystica_mask, mask)
        # cystica_mean, cystica_stddev = cv2.meanStdDev(dst, mask=cystica_mask)
        return ret, cystica_mask

    # 求图的梯度图，并且不归一化到0-255. 其中梯度等于x方向梯度和y方向梯度平方和开根号
    @staticmethod
    def compute_grad(img):
        gradX, gradY = np.gradient(img)
        final = (np.sqrt(gradX ** 2 + gradY ** 2) + 0.5).astype(int)
        return final

    # 非0-255灰度区域的图片中值滤波
    # TODO: 优化中值滤波，遍历的方法太慢
    @staticmethod
    def midFilter(img, ksize):
        kernel = np.ones((ksize, ksize)) / (ksize * ksize)
        img = cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
        return img


    @staticmethod
    def _erode(img, ksize):
        img = copy.deepcopy(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        img = cv2.erode(img, kernel)
        return img

    # 封装膨胀
    @staticmethod
    def _dilate(img, ksize):
        img = copy.deepcopy(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        img = cv2.dilate(img, kernel)
        return img

    ###############################################################################################
    # 形状描述工具，平滑，获取轮廓点，尺度归一化等
    ###############################################################################################

    @staticmethod
    def smooth_boundary(binary_shape, window_len=11, return_cnt=False):
        def smooth(x, window_len=11, window='hanning'):
            """smooth the data using a window with requested size.

            This method is based on the convolution of a scaled window with the signal.
            The signal is prepared by introducing reflected copies of the signal
            (with the window size) in both ends so that transient parts are minimized
            in the begining and end part of the output signal.

            input:
                x: the input signal
                window_len: the dimension of the smoothing window; should be an odd integer
                window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.

            output:
                the smoothed signal

            example:

            t=linspace(-2,2,0.1)
            x=sin(t)+randn(len(t))*0.1
            y=smooth(x)

            see also:

            numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
            scipy.signal.lfilter

            TODO: the window parameter could be the window itself if an array instead of a string
            NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
            """

            if x.ndim != 1:
                raise (ValueError, "smooth only accepts 1 dimension arrays.")

            if x.size < window_len:
                raise (ValueError, "Input vector needs to be bigger than window size.")

            if window_len < 3:
                return x

            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

            s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
            if window == 'flat':  # moving average
                w = np.ones(window_len, 'd')
            else:
                w = eval('np.' + window + '(window_len)')

            y = np.convolve(w / w.sum(), s, mode='valid')
            idx = int(window_len / 2)
            y = y[idx:-idx]
            return y

        SLIDING_WINDOW_LEN_RATIO = .02
        shape = copy.deepcopy(binary_shape)
        *_, contours, _ = cv2.findContours(shape,
                                       cv2.RETR_EXTERNAL,  # 只取外围轮廓
                                       cv2.CHAIN_APPROX_NONE)  # cv2.CHAIN_APPROX_**决定得到的轮廓点的数目，以及质量
        # 0814修改，对应多个cnt的情况，取最大的一个
        lens_list = [len(c) for c in contours]
        max_len_idx = np.argmax(lens_list)
        cnt = contours[max_len_idx]
        cnt = cnt.squeeze()
        x, y = cnt[:, 0], cnt[:, 1]

        sm_x = smooth(x, window_len=int(len(cnt) * SLIDING_WINDOW_LEN_RATIO * 2) + 1)
        sm_y = smooth(y, window_len=int(len(cnt) * SLIDING_WINDOW_LEN_RATIO * 2) + 1)

        sm_x = sm_x.astype(np.int)
        sm_y = sm_y.astype(np.int)

        sm_x = sm_x[:, np.newaxis]
        sm_y = sm_y[:, np.newaxis]
        smoooth_cnt = np.hstack((sm_x, sm_y))

        smooth_img = np.zeros(shape=binary_shape.shape, dtype=binary_shape.dtype)
        #     smooth_img = cv2.fillConvexPoly(smooth_img, smoooth_cnt, 255)
        # 填充多边形
        smooth_img = cv2.fillPoly(smooth_img,
                                  np.array([smoooth_cnt], dtype=np.int32),
                                  1)
        if return_cnt:
            return smooth_img, smoooth_cnt
        else:
            return smooth_img

    def classify_regular_shape(self):
        """
        计算两轴长度比，长轴/短轴，若比例大于1.1为类椭圆，否则类圆
        :return: regular_class, str, 类圆形或者椭圆形
        """
        proportion = max(self.hx_len, self.vx_len) / min(self.hx_len, self.vx_len)
        regualr_class = CIRCLE if proportion <= 1.1 else ELLIPSE
        return regualr_class

    def scale_normalize(self, binary_shape, L=128, threshold=.3):
        """
        对输入形状做尺度归一化，对于形状其余外接矩形（不是最小外接矩形框），将矩形框的长边缩放之128
        其短边等比例缩放。记录形状在原始图像上的位置与大小，方便返回原图显示
        :param binary_shape: 输入原始形状图像，numpy矩阵，
        :param L: 目标最长径，int，默认128
        :param threshold: 缩放之后二值化的阈值，float，默认0.3
        :return:
            zoom_shape 尺度归一化后的形状
            shape_meta 原始位置、缩放尺度信息等
        """
        cnt = self.get_contour(binary_shape)
        extend = 5
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect     # 轮廓最大矩形的外接
        # 是否边缘进行判断
        img_height = binary_shape.shape[0]
        img_width = binary_shape.shape[1]
        #  将边缘映射到归一到128后的坐标
        left_edge, right_edge, top_edge, bottom_edge = self.find_boud(cnt, img_height, img_width)
        max_edge_length = max(min(y + h + extend,binary_shape.shape[0]) - max(y - extend,0)+1, \
                              min(x + w + extend,binary_shape.shape[1]) - max(x - extend,0)+1)
        edge_scale = L / max_edge_length


        #  开始归一化。(边缘实际坐标-压缩矩形实际坐标)*比例尺
        left_edge = (left_edge - max(x - extend,0)) * edge_scale
        right_edge = (right_edge - min(x + w + extend,img_width)) * edge_scale + 128
        top_edge = (top_edge - max(y - extend,0)) * edge_scale
        bottom_edge = (bottom_edge - min(y + h + extend,img_height)) * edge_scale + 128
        x_change = max(x - extend,0)
        y_change = max(y - extend,0)


        

        max_side_length = max(h, w)
        zoom_scale = L / max_side_length
        rect_shape = binary_shape[max(y - extend,0):min(y + h + extend,binary_shape.shape[0]),\
                                  max(x - extend,0):min(x + w + extend,binary_shape.shape[1])]  # numpy x y 反的
        # rect = [max(y - extend,0), min(y + h + extend,binary_shape.shape[0]),\
        #         max(x - extend,0), min(x + w + extend,binary_shape.shape[1])]
        zoom_shape = cv2.resize(rect_shape,
                                (int(w * zoom_scale), int(h * zoom_scale)),
                                interpolation=cv2.INTER_CUBIC)
        max_ = zoom_shape.max()
        zoom_shape[zoom_shape > threshold * max_] = max_
        zoom_shape[zoom_shape < threshold * max_] = 0

        shape_meta = {
            'original shape': binary_shape.shape,
            'zoom scale': edge_scale,
            'rect': rect,
            'xy_change':(x_change, y_change)
        }
        return zoom_shape, shape_meta, [left_edge,right_edge,top_edge,bottom_edge]

    def find_boud(self, cnt, h, w):
        """
        找到与边缘相切割的边缘阈值
        """
        left_edge = 0
        right_edge = w
        top_edge = 0
        bottom_edge = h
        x = cnt[:,0]
        y = cnt[:,1]
        xmin_edge = (x < 5)
        xmax_edge = (x > w-5)
        ymin_edge = (y < 5) 
        ymax_edge = (y > h-5)
        if len(x[xmin_edge]) > 20:
            left_edge = 20
        if len(x[xmax_edge]) > 20:
            right_edge = w-20
        if len(y[ymin_edge]) > 20:
            top_edge = 20
        if len(y[ymax_edge]) > 20:
            bottom_edge = h-20
        return left_edge, right_edge, top_edge, bottom_edge

    @staticmethod
    def get_contour(binary_shape):
        """
        获取形状的轮廓，使用opencv的获取轮廓findcontour函数
        只获取最外围的轮廓点并保存所有点
        接下来会重新设置cnt的起始点，设置为质心坐标正上方的点，为起始点，逆时针开始
        :param binary_shape: 输入原始形状图像
        :return: cnt: 形状轮廓，numpy N*2 矩阵，保存轮廓点的xy坐标
        """

        def reset_start_point(cnt, centroid):
            """
            永远挑选centroid正上方最远点作为起始点
            方向：顺时针？
            """
            idx = np.argwhere(cnt[:, 0] == centroid[0])
            idx_ = np.argmin(cnt[idx, 1])
            idx = idx[idx_]
            idx = int(idx)
            new_cnt = np.vstack((cnt[idx:], cnt[:idx]))
            return new_cnt

        *_, contours, _ = cv2.findContours(copy.deepcopy(binary_shape),
                                       cv2.RETR_EXTERNAL,  # 只取外围轮廓
                                       cv2.CHAIN_APPROX_NONE)  # cv2.CHAIN_APPROX_**决定得到的轮廓点的数目
        length_list = []
        for c in contours:
            length_list.append(len(c))
        max_cnt_idx = np.argmax(length_list)
        cnt = contours[max_cnt_idx]
        cnt = cnt.squeeze()
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroid = (cx, cy)
        # 起点设置为质心正对的最上面的像素
        cnt = reset_start_point(cnt, centroid)
        return cnt

    @staticmethod
    def point_angle(cnt, pt2=None, pt3=None):
        """
        根据余弦定理计算三角形顶角角度
        :param cnt: 三角形第一个点坐标或者是三角形三个点坐标的点集，numpy数组
        :param pt2: 三角形第二个点坐标
        :param pt3: 三角形第三个点坐标
        :return: 中心点（顶角）的角度
        """
        if pt2 is not None and pt3 is not None:
            pt1 = cnt
        else:
            pt1, pt2, pt3 = cnt[0], cnt[1], cnt[2]
        x1, y1 = pt1[0], pt1[1]
        x2, y2 = pt2[0], pt2[1]
        x3, y3 = pt3[0], pt3[1]

        # 计算三条边长
        a = math.sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3))  # 2&3
        b = math.sqrt((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3))  # 1&3
        c = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))  # 1&2

        # 利用余弦定理计算三个角的角度
        # A = math.degrees(math.acos(round((a * a - b * b - c * c) / (-2 * b * c), 2)))
        B = math.degrees(math.acos(round((b * b - a * a - c * c) / (-2 * a * c), 2)))
        # C = math.degrees(math.acos(round((c * c - a * a - b * b) / (-2 * a * b), 2)))
        # A, B, C = round(A, 6), round(B, 6), round(C, 6)

        return round(B, 6)

    @staticmethod
    def create_shape(cnt, output_max=1, shape=(256, 256), dtype=np.uint8):
        """
        根据给定的点集cnt生成形状矩阵
        :param cnt:
        :param output_max:
        :param shape:
        :param dtype:
        :return:
        """
        img = np.zeros(shape=shape, dtype=dtype)
        cnt = np.array([cnt], dtype=cnt.dtype)
        img = cv2.fillPoly(img, cnt, 1, 255)
        # img[img > 0] = output_max
        return img


###############################################################################################
# 区域生长法
###############################################################################################
    @staticmethod
    def getGrayDiff(img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint.y, currentPoint.x]) - int(img[tmpPoint.y, tmpPoint.x]))

    @staticmethod
    def selectConnects(p):
        if p != 0:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),
                        Point(0, 1), Point(-1, 1), Point(-1, 0)]
        else:
            connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return connects

    def regionGrow(self, img, seeds, thresh, mask, p=1):
        height, weight = img.shape
        seedMark = np.zeros(img.shape, np.uint8)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        label = 255
        connects = self.selectConnects(p)
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)

            seedMark[currentPoint.y, currentPoint.x] = label
            for i in range(8):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpY >= height or tmpX >= weight:
                    continue
                if mask[tmpY, tmpX] == 0:
                    continue
                # flag = abs(img[tmpY, tmpX] - low_gray) / abs(img[tmpY, tmpX] - high_gray)
                if img[tmpY, tmpX] > thresh:
                    continue
                # for j in range(8):
                #     tmpX2 = tmpX + connects[j].x
                #     tmpY2 = tmpY + connects[j].y
                #     if tmpX2 < 0 or tmpY2 < 0 or tmpY2 >= height or tmpX2 >= weight:
                #         continue
                #     if mask[tmpY2, tmpX2] == 0:
                #         continue
                #     if img[tmpY2, tmpX2] != thresh:
                #         break
                #     if i == 7 and seedMark[tmpY, tmpX] == 0:
                #         seedMark[tmpY, tmpX] = label
                #         seedList.append(Point(tmpX, tmpY))
                if i == 7 and seedMark[tmpY, tmpX] == 0:
                    seedMark[tmpY, tmpX] = label
                    seedList.append(Point(tmpX, tmpY))
        return seedMark
    ########################################################################################
    # 区域生长法结束
    ########################################################################################


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


class CurvatureCalculator(object):
    """
    曲率计算类，
    对于一个形状的轮廓的曲率计算方法：
    遍历每一个轮廓点，去前后N个点的X Y坐标，分别拟合x(t) y(t)函数，通过曲率计算公式计算这一点的曲率
    """

    def __init__(self):
        pass

    def find_zero_crossings(self, kappa):
        """ 寻找一个信号的零值点
        零值点定义为前一个点与后一个点正负符号不相符的点
        """
        crossings = []
        for i in range(0, kappa.size - 2):
            if (kappa[i] < 0.0 and kappa[i + 1] > 0.0) or (kappa[i] > 0.0 and kappa[i + 1] < 0.0):
                crossings.append(i)
        return crossings

    def gaussian_smooth(self, x, sigma, range_=3, plot=False):
        """对输入信号（xy坐标，分别）做高斯平滑
        Args:
            :param x:输入信号
            :param sigma:平滑参数
            :param range_:相当于range_*sigma为高斯核的大小

        """
        total_len = len(x)
        X = np.zeros(shape=(total_len,))
        for i, xi in enumerate(x):
            new_xi = 0  # 点i的新数值，是一个累加2*range_+1的累加数值
            miu = 0  # 中心位置，在这里应该指的就是中心点，
            g = []  # 为了检查得到的高斯核函数做的
            for v in np.mgrid[-range_:range_ + 1]:
                e = math.exp(-(miu - v) ** 2 / (2 * sigma ** 2))
                e = e * (1 / (sigma * math.sqrt(math.pi * 2)))
                new_xi = new_xi + x[(i + v) % total_len] * e
                #         g.append(e)    # 把2*range_+1个exp的值拿出来看看
            #         print('old x:{0}'.format(x[i+v]), e, -(miu-v), new_xi)
            X[i] = new_xi
        return X

    def arc_by_N(self, cnt, i, n=10):
        half = int(n / 2)
        index = i
        if index - half < 0 and index + half > 0:
            arc_pts = np.vstack((cnt[index - half:], cnt[0:index + half]))
        else:
            arc_pts = cnt[index - half:index + half]

        arc_x = arc_pts[:, 0]
        arc_y = arc_pts[:, 1]

        return arc_x, arc_y

    def curvature(self, cnt, N=25, degree=2):
        """
        计算一个二维形状的外边缘曲率。
        边缘点高斯平滑!!
        虽然边缘点的坐标可能不是整数了
，但是还是和整数的颁布一一对应的。
        1.提取边缘，使用cv2.CHAIN_APPROX_SIMPLE或cv2.CHAIN_APPROX_NONE保存重要的边缘
        2.遍历边缘点，对于每个点
            a.取前后共N个点拟合多项式曲线，计算这个点的弧长
            b.以点为中心，取满足弧长=arclen的一段点集，同样拟合曲线，求曲率
            c.平滑曲率，得到最后的结果
        Args：
            :param shape： numpy.matrix, 输入图像（二值图）
            :param N: int,default 15，连续选择N个点作为弧
            :param degree: int，default 2，选择拟合弧长多项式的阶
            :param plot: bool, default False，是否展示结果

        Output:
            :return curvature: numpy array,形状轮廓曲率

        """
        mid = int(N / 2)
        curvature = np.zeros(shape=(len(cnt),))
        for i in range(len(cnt)):
            arc_x, arc_y = self.arc_by_N(cnt, i, N)
            t = np.mgrid[0:len(arc_x)]

            # x方向
            poly_x = np.polyfit(t, arc_x, deg=degree)
            p_x = np.polynomial.Polynomial(poly_x[::-1])
            p_x1 = p_x.deriv(1)
            p_x2 = p_x.deriv(2)

            # y 方向
            poly_y = np.polyfit(t, arc_y, deg=degree)
            p_y = np.polynomial.Polynomial(poly_y[::-1])
            p_y1 = p_y.deriv(1)
            p_y2 = p_y.deriv(2)

            K = (p_x1(mid) * p_y2(mid)) - (p_y1(mid) * p_x2(mid))
            cv = K / (p_x1(mid) ** 2 + p_y1(mid) ** 2) ** 1.5
            cv = round(cv, 4)
            curvature[i] = cv

        return curvature


# def shape_class():
#     temp_mask = r'D:\Users\XULANG197\Pictures\thyroid_data\TNSCUI2020_train\TNSCUI_train_600_17600'
#     GT_path = r'D:\Users\XULANG197\Pictures\thyroid_data\TNSCUI2020_train\mask'
#     temp_list = os.listdir(temp_mask)
#     regular_image_path = r'D:\Users\XULANG197\Pictures\thyroid_data\TNSCUI2020_train\TNSCUI_train_600_17600'
#     irregular_image_path = r'D:\Users\XULANG197\Pictures\thyroid_data\shape_classification\irregular_mask'

#     regular_list = os.listdir(regular_image_path)
#     irregular_list = os.listdir(irregular_image_path)
#     img_count = 0
#     for name in temp_list:
#         if name.endswith('_r.png'):
#             mask_name = name
#         else:
#             continue
#         #mask_name = "test_103_r.png"
#         image_name = mask_name[:-6] + '.PNG'
#         GT_name = mask_name[:-6] + '_c_.PNG'
#         GT = cv2.imread(os.path.join(GT_path, GT_name),0)
#         image_gray = cv2.imread(os.path.join(temp_mask, image_name),0)
#         if image_gray is None:
#             continue
#         print(mask_name)
#         mask = cv2.imread(os.path.join(temp_mask, mask_name),0)
#         mask[mask > 127] = 255
#         mask[mask != 255] = 0

#         img_count += 1
#         if img_count == 1:
#             tirads = TiradsRecognition(image_gray, mask, image_name, is_debug=False)
#         else:
#             tirads.img = image_gray
#             tirads.mask = mask
#             tirads.image_name = image_name

#         # 3.4 shape
#         shape = tirads.classify_shape()

#         folder_path = r"D:\Users\XULANG197\Pictures\thyroid_data\shape_classification\results"
#         new_regular_path = os.path.join(folder_path,'regular')
#         new_irregular_path = os.path.join(folder_path,'irregular')
#         if not os.path.exists(new_regular_path):
#             os.makedirs(new_regular_path)
#         if not os.path.exists(new_irregular_path):
#             os.makedirs(new_irregular_path)

#         if shape == 1:
#             cv2.imwrite(os.path.join(new_irregular_path, mask_name), mask)
#             cv2.imwrite(os.path.join(new_irregular_path, GT_name), GT)
#         elif shape == 0:
#             cv2.imwrite(os.path.join(new_regular_path, mask_name),mask)  
#             cv2.imwrite(os.path.join(new_regular_path, GT_name), GT) 

    # for name in temp_list:
    #     if name.endswith('_r.jpg'):
    #         mask_name = name
    #     else:
    #         continue
        
    #     #mask_name = "Mindray_600485513-20160623-US-1-6_c_s0_A6626_L121_PTC_m.jpg"
    #     print(mask_name)
    #     image_name = mask_name[:-6] + '.jpg'
    #     image_gray = cv2.imread(os.path.join(irregular_image_path, image_name),0)
    #     mask = cv2.imread(os.path.join(irregular_image_path, mask_name),0)
    #     mask[mask > 127] = 255
    #     mask[mask != 255] = 0

    #     img_count += 1
    #     if img_count == 1:
    #         tirads = TiradsRecognition(image_gray, mask, image_name, is_debug=False)
    #     else:
    #         tirads.img = image_gray
    #         tirads.mask = mask
    #         tirads.image_name = image_name
            
    #     # 3.4 shape
    #     shape = tirads.classify_shape()

    #     folder_path = r"D:\Users\XULANG197\Pictures\thyroid_data\shape_classification\results"
    #     new_TP_path = os.path.join(folder_path,'TP')
    #     new_FN_path = os.path.join(folder_path,'FN')
    #     if not os.path.exists(new_TP_path):
    #         os.makedirs(new_TP_path)
    #     if not os.path.exists(new_FN_path):
    #         os.makedirs(new_FN_path)

    #     if shape == 1:
    #         cv2.imwrite(os.path.join(new_TP_path, mask_name), mask)
    #     elif shape == 0:
    #         cv2.imwrite(os.path.join(new_FN_path, mask_name),mask)   

if __name__ == '__main__':
    shape_class()
    exit
    image_path = r'D:\Users\XULANG197\Pictures\thyroid_data\TNSCUI2020_train\JPEGImages'
    mask_path = r'D:\Users\XULANG197\Pictures\thyroid_data\TNSCUI2020_train\TNSCUI_shape_results\all_image'
    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    img_count = 0
    for mask_name in mask_list:
        mask_name = "285_c_.PNG"
        print(mask_name)
        image_name = mask_name[:-7] + mask_name[-4:]
        image_gray = cv2.imread(os.path.join(image_path, image_name), 0)
        mask = cv2.imread(os.path.join(mask_path, mask_name),0)
        mask[mask > 127] = 255
        mask[mask != 255] = 0

        img_count += 1
        if img_count == 1:
            tirads = TiradsRecognition(image_gray, mask, is_debug=False)
        else:
            tirads.img = image_gray
            tirads.mask = mask

        # 3.4 shape
        shape = tirads.classify_shape()

        folder_path = r"D:\Users\XULANG197\Pictures\thyroid_data\TNSCUI2020_train\TNSCUI_shape_results\v3.2_new_angle&thresh"
        new_regular_path = os.path.join(folder_path,'regular')
        new_irregular_path = os.path.join(folder_path,'irregular')
        if not os.path.exists(new_regular_path):
            os.makedirs(new_regular_path)
        if not os.path.exists(new_irregular_path):
            os.makedirs(new_irregular_path)

        if shape == 1:
            cv2.imwrite(os.path.join(new_irregular_path, mask_name), mask)
        elif shape == 0:
            cv2.imwrite(os.path.join(new_regular_path, mask_name),mask)


