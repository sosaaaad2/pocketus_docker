# -*- coding: utf-8 -*-
"""The Python implementation of the GRPC mrcnn server."""

import argparse
from concurrent import futures
import logging
import logging.handlers
import os.path
import time

import cv2
from google.protobuf.any_pb2 import Any
import grpc
import thyroidrpc_pb2
import thyroidrpc_pb2_grpc

from status import *
import numpy as np
#import rescnn
import Thyroid_class as rescnn
#import pymrcnn
import yolov3_pytorch.interface as pymrcnn
import yolov3_pytorch.interface_part as pymrcnn_part
#import LabelingSoftware_MaskRCNN_2ndSeg as maskrcnn2seg
import Thyroid_BESNET as maskrcnn2seg
import Thyroid_diffuse as diffuse
#from tirads import TiradsRecognition
from tirads_new.tirads_for_pocketus import TiradsRecognition
import userdb
import pymongo

#import importlib
#importlib.reload(sys)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# argument parser setup
parser = argparse.ArgumentParser(description='thyroid ai server.')

parser.add_argument(
    '-p',
    '--port',
    type=str,
    default='50051',
    #default='27017'
    required=False,
    help="set listening port, default is 50051."
)

# TODO: use other user friendly parameter
parser.add_argument(
    '--level',
    type=int,
    default=logging.INFO,
    required=False,
    help="set logging level, default is INFO."
)

parser.add_argument(
    '-log',
    '--logpath',
    type=str,
    default='/root/log',
    required=False,
    help="set logging path."
)

parser.add_argument(
    '--mongo',
    type=str,
    ##default='localhost:27017',
    default='172.17.0.1:27017', 
    required=False,
    help="set mongodb server ip and port!"
)

args = parser.parse_args()

class ThyroidaiGrpc(thyroidrpc_pb2_grpc.ThyroidaiGrpcServicer):

    def __init__(self):
        try:
            rescnn.model_init()
            logging.info('rescnn model init ok!')
        except:
            logging.critical('rescnn model init failed')
            raise Exception('rescnn model init failed')

        try:
            maskrcnn2seg.model_init()
            logging.info('BESNet model init ok!')
        except:
            logging.critical('BESNet model init failed')
            raise Exception('BESNet model init failed')

        try:
            pymrcnn_part.ThyroidAIServerInit()
            logging.info('yolo_part model init ok!')
        except:
            logging.critical('yolo_part model init failed')
            raise Exception('yolo_part model init failed')

        try:
            diffuse.model_init()
            logging.info('diffuse model init ok!')
        except:
            logging.critical('diffuse model init failed')
            raise Exception('diffuse model init failed')

        try:
            pymrcnn.ThyroidAIServerInit()  # (batch_size, gpu, bserilize)
            logging.info('yolo init ok')
        except:
            logging.critical('yolo init failed!')
            raise Exception('yolo init failed!')

        try:
            host, port = args.mongo.split(':')
            userdb.init_mongodb(host, int(port))
            logging.info('user database init ok')
        except:
            logging.critical('user database init failed!')
            raise Exception('user database init failed!')
    
        # do warmup
        self._DoWarmup()
        logging.info('do warmup ok!')

    # TODO: warmup maskrcnn tensorrt module, waiting input image format changing to gray!
    # warmup
    def _DoWarmup(self):
        image = cv2.imread('../images/03.jpg')
        gray = image[:,:,1]
        h, w = gray.shape
        #gray = gray.reshape([gray.size])
        mask = cv2.imread('../images/03_mask.jpg', 0)
        for _i in range(2):
            print(pymrcnn.ThyroidAIDoInference(gray, h, w))
            print(maskrcnn2seg.do_inference(gray, (410, 179, 545, 302)))
            print(rescnn.do_inference(gray, mask))
            anno, c_s =pymrcnn_part.ThyroidAIDoInference(gray, h, w)
            print(anno, c_s)
            print(diffuse.inference(gray,anno))

    def __del__(self):
        #pymrcnn.ThyroidAIServerFinalize()
        pass

    def ImageOutput(self, request, context):       
        
        height = request.height
        width = request.width
        isRaw = request.isRaw
        print('image processing')
        image = np.frombuffer(request.image, dtype=np.uint8)
        ppc = request.ppc
        
        if not request.isRaw:
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            height, width = image.shape
            #image = image.reshape([height,width])
        
        # check image size, single-channel gray image
        if 0 == image.size or image.size != height * width:
            logging.error(f'image shape is {image.shape}, but height = {height}, width = {width}')
            return thyroidrpc_pb2.ProtoResponse(code=WRONG_IMAGE_SHAPE,
                                                msg=ErrorDict[WRONG_IMAGE_SHAPE],
                                                data=anypb) 
        if image.ndim == 1:
            image = image.reshape([height,width])
        response = self.Detect(thyroidrpc_pb2.DetectRequest(isRaw=isRaw, image=request.image, height=height, width=width), 1)
        if (response.code !=0 ):
            print('error code: {}'.format(response.code))
            print('error message: {}'.format(response.msg))
        else:
            anypb = Any()
            anypb.CopyFrom(response.data)

            nodules = thyroidrpc_pb2.Nodules()
            anypb.Unpack(nodules)

            print(f'nodule number is: {nodules.nums}')
            for node in nodules.nodule:
                print(f'nudule {node.n}: ({node.x}, {node.y}, {node.w}, {node.h})')

        ###########################################
        ### 2. test second stage inference      ###
        ###########################################

        user_id = '123456'
        scan_id = int(time.time()*10)
        image_id = 1

        list_nodule = []
        isPaw = True

        cut = 1
        for node in nodules.nodule:
            nx, ny, nw, nh = (node.x, node.y, node.w, node.h)
            list_nodule.append(thyroidrpc_pb2.NoduleWithNum(n=node.n, m=0, x=nx, y=ny, w=nw, h=nh, s=cut, pos=5))
            if cut == 1:
                cut = 2
            elif cut == 2:
                cut = 1
            else:
                pass
            
        if len(list_nodule) > 0:
            image_gray, list_str = self.DetectImage(
                    thyroidrpc_pb2.CTRequest(
                        uid=user_id,
                        scanID=scan_id,
                        imageID=image_id,
                        isRaw=isRaw,
                        image=request.image, 
                        height=height, 
                        width=width, 
                        ppc=ppc, 
                        nodule=list_nodule
                    ), 1)
        else:
            image_gray = image
            read = "未检测到结节"
            list_str = [read.encode(encoding="utf-8")]
        if (response.code !=0 ):
            print('error code: {}'.format(response.code))
            print('error message: {}'.format(response.msg))
        else:
            anypb = Any()
            anypb.CopyFrom(response.data)

            ctRes = thyroidrpc_pb2.CTResponse()
            anypb.Unpack(ctRes)
        
        img=image_gray.tostring()
        print(list_str)
        anypb.Pack(thyroidrpc_pb2.ImageRequest(image=img, msg = list_str))
        return thyroidrpc_pb2.ProtoResponse(code=0, msg='', data=anypb)
        
        

    def DetectImage(self, request, context):
        """ classification, tirads
        """

        anypb = Any()
        image_gray = np.frombuffer(request.image, dtype=np.uint8)
        height = request.height
        width = request.width

        if not request.isRaw:
            image_gray = cv2.imdecode(image_gray, cv2.IMREAD_GRAYSCALE)
            height, width = image_gray.shape

        # check image size
        if 0 == image_gray.size or image_gray.size != height * width:
            logging.error(f'image shape is {image_gray.shape}, but height = {height}, width = {width}')
            return thyroidrpc_pb2.ProtoResponse(code=WRONG_IMAGE_SHAPE,
                                                msg=ErrorDict[WRONG_IMAGE_SHAPE],
                                                data=anypb)

        if image_gray.ndim == 1:
            image_gray = image_gray.reshape([height, width])

        logging.info(f'image shape is height = {height}, width = {width}')

        # gray to rgb
        image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
 
        # check input nodule number
        if len(request.nodule) <= 0:
            logging.error(f'input nodule numbers should large than zero, but your input is: {len(request.nodule)}')
            return thyroidrpc_pb2.ProtoResponse(code=WRONG_INPUT_NODELES,
                                                msg=ErrorDict[WRONG_INPUT_NODELES],
                                                data=anypb)

        # 3.0 diffuse, c_s
        try:
            anno, c_s =pymrcnn_part.ThyroidAIDoInference(image_gray, height, width)
        except:
            logging.error('part detection fail!')
            return thyroidrpc_pb2.ProtoResponse(code=TIRADS_FAIL,
                                                msg=ErrorDict[CLASSIFICATION_FAIL],
                                                data=anypb)

        # output list
        list_ds = []
        list_nodule = []
        list_benign_prob = []
        list_tirads = []
        list_str = []
        

        # loop over all the nodules
        for nodule in request.nodule:

            t1 = time.perf_counter()

            logging.info(f'nodule index: {nodule.n}, pos = ({nodule.x}, {nodule.y}, {nodule.w}, {nodule.h})')
            # 1. do second stage segmentation
            bbox_input = (nodule.x, nodule.y, nodule.x + nodule.w, nodule.y + nodule.h)

            try:
                mask, bbox = maskrcnn2seg.do_inference(image_gray, bbox_input)
            except:
                logging.error('second stage detection exception')
                return thyroidrpc_pb2.ProtoResponse(code=SECOND_DETECTION_FAIL,
                                                    msg=ErrorDict[SECOND_DETECTION_FAIL],
                                                    data=anypb)
            if mask.size == 0:
                list_ds.append(thyroidrpc_pb2.DetectionStatus(n=nodule.n, status=1))
                list_nodule.append([])
                list_benign_prob.append([])
                list_tirads.append([])
                continue
            else:
                list_nodule.append(thyroidrpc_pb2.NoduleWithNum(n=nodule.n, m=nodule.m, x=bbox[0], y=bbox[1], 
                                                                w=bbox[2]-bbox[0], h=bbox[3]-bbox[1], s=nodule.s, pos=nodule.pos))

            bbox_pos = [bbox[0].item(), bbox[1].item(), (bbox[2]-bbox[0]).item(), (bbox[3]-bbox[1]).item()]

            t2 = time.perf_counter()
            logging.info('nodule {}, second detection time: {:.2f}'.format(nodule.n, (t2-t1)*1000))

            # 2. do classification
            t1 = time.perf_counter()

            try:
                benign, prob = rescnn.do_inference(image_gray, mask)
            except:
                logging.error('classification fail!')
                return thyroidrpc_pb2.ProtoResponse(code=CLASSIFICATION_FAIL,
                                                    msg=ErrorDict[CLASSIFICATION_FAIL],
                                                    data=anypb)
 
            # 1: benign; 2: not benign
            benign += 1
            list_benign_prob.append(thyroidrpc_pb2.BenignAndProb(benign=benign, prob=prob))
            t2 = time.perf_counter()
            logging.info('nodule {}, classification time: {:.2f}'.format(nodule.n, (t2-t1)*1000))

            # 3. do tirads
            """
            constitute 构成  0： 实性    1： 实性为主  2： 囊性为主  3： 囊性
            comet 彗星尾数量， 不过不要显示数量，直接大于1时显示有彗星尾就行
            shape 0： 形状规则  1： 形状不规则
            ar  # 纵横比
            hx_len  # 水平轴 如果是横切，它就是左右径，如果是纵切，它就是上下径
            vx_len  # 垂直轴  它只有可能是前后径
            echo_code  # 回声水平  0： 无回声  2： 低回声  3： 等回声  4： 高回声
            border_clear  # 边界清晰模糊   0： 模糊  1： 清晰
            calcification[0] # 0：无微钙化  1： 有微钙化
            calcification[1] # 0：无粗钙化  1： 有粗钙化
            calcification[2] # 0：无环钙化  1： 有环钙化
            """


            
            try:
                #print(nodule.x, nodule.y, nodule.w, nodule.h)
                if nodule.w*nodule.h < 100000:
                    diffusion = diffuse.inference(image_gray,anno)
                else:
                    diffusion = 2
            except:
                logging.error('diffuse fail!')
                return thyroidrpc_pb2.ProtoResponse(code=TIRADS_FAIL,
                                                    msg=ErrorDict[CLASSIFICATION_FAIL],
                                                    data=anypb)

            t1 = time.perf_counter()
            tirads = TiradsRecognition(image_gray, mask, is_debug=False)

            # 3.1 constitute
            constitute, cys_mask = tirads.find_or_report_constitute("report")

            # 3.2 comet
            if constitute <= 1:
                comet = 0
            else:
                comet = tirads.find_comet_calcify("report", cys_mask)

            # 3.3 ratio and hx_len, vx_len
            ar, vx, hx = tirads.estimate_aspect_ratio()

            hx_len, vx_len = tirads.get_axis_len(hx), tirads.get_axis_len(vx)

            pixels_one_cm = request.ppc

            hx_len /= pixels_one_cm
            vx_len /= pixels_one_cm
            hx_len = round(hx_len, 8)
            vx_len = round(vx_len, 8)

            if nodule.s == 1:
                hx_list = [hx_len, 0]
            else:
                hx_list = [0, hx_len]


            # 3.4 shape
            shape = tirads.classify_shape()

            # TODO: echo lever
            # 3.5 echo level
            #echo_code = tirads.compute_nodule_echo(mask_cyst=cys_mask)
            echo_code = 2

            # 3.6 border
            border_clear = tirads.border_clear_fuzzy()

            # 3.7 calcification 钙化
            if comet > 0:
                calcification = tirads.find_or_report_calcification(pixels_one_cm)
            else:
                calcification = [0, 0, 0]
            
            calcification.append(diffusion)

            list_tirads.append(
                thyroidrpc_pb2.OneTiradsRes(constitute = constitute, 
                                            comet = comet,
                                            shape = shape,
                                            ratio = ar,
                                            hxlen = hx_list,
                                            vxlen = vx_len,
                                            echo_level = echo_code,
                                            border_clear = border_clear,
                                            calcification = calcification
                                            )
            )

            # at last we set status ok
            list_ds.append(thyroidrpc_pb2.DetectionStatus(n=nodule.n, status=0))

            # draw mask in original image
            ret, dst = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            contours,hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image_gray, contours, -1, (230, 255, 0), 1)
            
            x,y,w,h=bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
            cv2.putText(image_gray, 'nodule ' + str(nodule.n), (int(x+w/2-25), y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,255,0), 2)

            nodule_read = np.ones((30, width))
            if c_s == 1:
                str_c_s = "横切,"
                str_hx_len = "左右径"+ str(hx_len) + "cm,"
            elif c_s== 0:
                str_c_s = "纵切,"
                str_hx_len = "上下径"+ str(hx_len) + "cm,"
                
            str_vx_len = "前后径"+ str(vx_len) + "cm,"
            str_ar = "纵横比" + str(ar) + "," 
                
            if diffusion == 1:
                str_diffusion = "弥漫性,"
            elif diffusion == 0:
                str_diffusion = "非弥漫性,"
            elif diffusion == 2:
                str_diffusion = ""
            
            if benign == 1:
                str_beni = "良性,"
            elif benign == 2:
                str_beni = "恶性,"

            if constitute == 0:
                str_constitute = "实性,"
            elif constitute == 1:
                str_constitute = "实性为主,"   
            elif constitute == 2:
                str_constitute = "囊性为主,"
            elif constitute == 3:
                str_constitute = "囊性," 

            if comet > 0:
                str_comet = "有彗星尾,"
            else:
                str_comet = "无彗星尾,"

            if shape == 0:
                str_shape = "形状规则,"
            elif shape == 1:
                str_shape = "形状不规则,"

            if border_clear == 0:
                str_echo = "模糊,"
            elif border_clear == 1:
                str_echo = "清晰,"

            if calcification[0] == 0:
                str_calc0 = "无微钙化,"
            elif calcification[0] == 1:
                str_calc0 = "有微钙化,"
            if calcification[1] == 0:
                str_calc1 = "无粗钙化,"
            elif calcification[1] == 1:
                str_calc1 = "有粗钙化,"
            if calcification[2] == 0:
                str_calc2 = "无环钙化,"
            elif calcification[2] == 1:
                str_calc2 = "有环钙化,"

            read ='nodule ' + str(nodule.n) + ": " + str_beni + str_c_s + str_diffusion + str_constitute + str_comet + str_shape + str_echo + str_calc0 + str_calc1 + str_calc2 + str_hx_len + str_vx_len + str_ar
            read = read.encode(encoding="utf-8")
     
            list_str.append(read)
        print("image done")
        #print(len(list_str[0]))

        return image_gray, list_str

    def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if (isinstance(img, numpy.ndarray)):  #判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype(
            "font/simsun.ttc", textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontText)
        return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    def Detect(self, request, context):
        """ real time detection task!
        """
        t1 = time.perf_counter()
        anypb = Any()

        height = request.height
        width = request.width

        image = np.frombuffer(request.image, dtype=np.uint8)
        #if image.ndim == 1:
        #        image = image.reshape([height,width])
        if not request.isRaw:
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            height, width = image.shape
            image = image.reshape([height,width])

        # check image size, single-channel gray image
        if 0 == image.size or image.size != height * width:
            logging.error(f'image shape is {image.shape}, but height = {height}, width = {width}')
            return thyroidrpc_pb2.ProtoResponse(code=WRONG_IMAGE_SHAPE,
                                                msg=ErrorDict[WRONG_IMAGE_SHAPE],
                                                data=anypb) 

        if image.ndim == 1:
            image = image.reshape([height,width])
        res = pymrcnn.ThyroidAIDoInference(image, height, width)
        print("res:", res)
        nodule_nums = len(res)
        logging.info(f'nodules_nums: {nodule_nums}')
        print("nodule_nums:" + str(nodule_nums))
        if 0 == nodule_nums:
            nodules = thyroidrpc_pb2.Nodules(nums=0, nodule=[])
        else:
            nodules_list = []
            for i in range(nodule_nums):
                nodules_list.append(
                    thyroidrpc_pb2.NoduleWithNum(n=i+1, m=0, x=res[i][0], y=res[i][1], w=res[i][2], h=res[i][3], s=0, pos=0)
                    )
            nodules = thyroidrpc_pb2.Nodules(nums=nodule_nums, nodule=nodules_list)
        print("nodules:",nodules.nums, nodules.nodule)
        # packing results to any
        anypb.Pack(nodules)
        t2 = time.perf_counter()
        logging.info('tensorrt: {:.2f}'.format((t2-t1)*1000))
        print(nodules)
        return thyroidrpc_pb2.ProtoResponse(code=0, msg='', data=anypb)

    def ClassifyAndTirads(self, request, context):
        """ classification, tirads
        """

        logging.info('---------------------------------------')
        logging.info('ClassifyAndTirads')
        logging.info(f'uid: {request.uid}, scanID: {request.scanID}, imageID: {request.imageID}')

        anypb = Any()

        image_gray = np.frombuffer(request.image, dtype=np.uint8)
        height = request.height
        width = request.width

        if not request.isRaw:
            image_gray = cv2.imdecode(image_gray, cv2.IMREAD_GRAYSCALE)
            height, width = image_gray.shape

        # check image size
        if 0 == image_gray.size or image_gray.size != height * width:
            logging.error(f'image shape is {image_gray.shape}, but height = {height}, width = {width}')
            return thyroidrpc_pb2.ProtoResponse(code=WRONG_IMAGE_SHAPE,
                                                msg=ErrorDict[WRONG_IMAGE_SHAPE],
                                                data=anypb)

        if image_gray.ndim == 1:
            image_gray = image_gray.reshape([height, width])

        logging.info(f'image shape is height = {height}, width = {width}')

        # gray to rgb
        image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
 
        # check input nodule number
        if len(request.nodule) <= 0:
            logging.error(f'input nodule numbers should large than zero, but your input is: {len(request.nodule)}')
            return thyroidrpc_pb2.ProtoResponse(code=WRONG_INPUT_NODELES,
                                                msg=ErrorDict[WRONG_INPUT_NODELES],
                                                data=anypb)
        """
        what's the logic?
        1. loop over request bounding box, use second stage maskrcnn to generate new mask
        2. for each new mask/nodule, do classification task
        3. for each new mask/nodule, do tirads task
        4. save the results for final report
        """

        # output list
        list_ds = []
        list_nodule = []
        list_benign_prob = []
        list_tirads = []
        modify_flag = []

        # loop over all the nodules
        for nodule in request.nodule:
            modify_flag.append(False)
            t1 = time.perf_counter()

            logging.info(f'nodule index: {nodule.n}, pos = ({nodule.x}, {nodule.y}, {nodule.w}, {nodule.h})')
            # 1. do second stage segmentation
            bbox_input = (nodule.x, nodule.y, nodule.x + nodule.w, nodule.y + nodule.h)

            try:
                mask, bbox = maskrcnn2seg.do_inference(image_gray, bbox_input)
                print(bbox)
            except:
                logging.error('second stage detection exception')
                return thyroidrpc_pb2.ProtoResponse(code=SECOND_DETECTION_FAIL,
                                                    msg=ErrorDict[SECOND_DETECTION_FAIL],
                                                    data=anypb)
            if mask.size == 0:
                logging.error(f'{nodule.n}: second stage not detect any nodule!')
                list_ds.append(thyroidrpc_pb2.DetectionStatus(n=nodule.n, status=1))
                list_nodule.append([])
                list_benign_prob.append([])
                list_tirads.append([])
                continue
            else:
                list_nodule.append(thyroidrpc_pb2.NoduleWithNum(n=nodule.n, m=nodule.m, x=bbox[0], y=bbox[1], 
                                                                w=bbox[2]-bbox[0], h=bbox[3]-bbox[1], s=nodule.s, pos=nodule.pos))

            bbox_pos = [bbox[0].item(), bbox[1].item(), (bbox[2]-bbox[0]).item(), (bbox[3]-bbox[1]).item()]

            t2 = time.perf_counter()
            logging.info('nodule {}, second detection time: {:.2f}'.format(nodule.n, (t2-t1)*1000))

            # 2. do classification
            t1 = time.perf_counter()
            
            try:
                benign, prob = rescnn.do_inference(image_gray, mask)
                print(benign, prob)
            except:
                logging.error('classification fail!')
                return thyroidrpc_pb2.ProtoResponse(code=CLASSIFICATION_FAIL,
                                                    msg=ErrorDict[CLASSIFICATION_FAIL],
                                                    data=anypb)
 
            # 1: benign; 2: not benign
            benign += 1
            list_benign_prob.append(thyroidrpc_pb2.BenignAndProb(benign=benign, prob=prob))
            t2 = time.perf_counter()
            logging.info('nodule {}, classification time: {:.2f}'.format(nodule.n, (t2-t1)*1000))

            # 3. do tirads
            """
            constitute 构成  0： 实性    1： 实性为主  2： 囊性为主  3： 囊性
            comet 彗星尾数量， 不过不要显示数量，直接大于1时显示有彗星尾就行
            shape 0： 形状规则  1： 形状不规则
            ar  # 纵横比
            hx_len  # 水平轴 如果是横切，它就是左右径，如果是纵切，它就是上下径
            vx_len  # 垂直轴  它只有可能是前后径
            echo_code  # 回声水平  0： 无回声  2： 低回声  3： 等回声  4： 高回声
            border_clear  # 边界清晰模糊   0： 模糊  1： 清晰
            calcification[0] # 0：无微钙化  1： 有微钙化
            calcification[1] # 0：无粗钙化  1： 有粗钙化
            calcification[2] # 0：无环钙化  1： 有环钙化
            """

            # 3.0 diffuse, c_s
            
            try:
                anno, c_s =pymrcnn_part.ThyroidAIDoInference(image_gray, height, width)
                print(anno, c_s)
            except:
                logging.error('part detection fail!')
                return thyroidrpc_pb2.ProtoResponse(code=TIRADS_FAIL,
                                                    msg=ErrorDict[CLASSIFICATION_FAIL],
                                                    data=anypb)
            
            try:
                if nodule.w*nodule.h < 100000:
                    diffusion = diffuse.inference(image_gray,anno)
                else:
                    diffusion = 2
            except:
                logging.error('diffuse fail!')
                return thyroidrpc_pb2.ProtoResponse(code=TIRADS_FAIL,
                                                    msg=ErrorDict[CLASSIFICATION_FAIL],
                                                    data=anypb)

            t1 = time.perf_counter()
            tirads = TiradsRecognition(image_gray, mask, is_debug=False)

            # 3.1 constitute
            constitute, cys_mask = tirads.find_or_report_constitute("report")

            # 3.2 comet
            if constitute <= 1:
                comet = 0
            else:
                comet = tirads.find_comet_calcify("report", cys_mask)

            # 3.3 ratio and hx_len, vx_len
            ar, vx, hx = tirads.estimate_aspect_ratio()

            hx_len, vx_len = tirads.get_axis_len(hx), tirads.get_axis_len(vx)

            pixels_one_cm = request.ppc

            hx_len /= pixels_one_cm
            vx_len /= pixels_one_cm
            hx_len = round(hx_len, 8)
            vx_len = round(vx_len, 8)

            if nodule.s == 1:
                hx_list = [hx_len, 0]
            else:
                hx_list = [0, hx_len]


            # 3.4 shape
            shape = tirads.classify_shape()

            # TODO: echo lever
            # 3.5 echo level
            #echo_code = tirads.compute_nodule_echo(mask_cyst=cys_mask)
            echo_code = 2

            # 3.6 border
            border_clear = tirads.border_clear_fuzzy()

            # 3.7 calcification 钙化
            if comet > 0:
                calcification = tirads.find_or_report_calcification(pixels_one_cm)
            else:
                calcification = [0, 0, 0]
            # 加入弥漫性结果
            calcification.append(diffusion)

            list_tirads.append(
                thyroidrpc_pb2.OneTiradsRes(constitute = constitute, 
                                            comet = comet,
                                            shape = shape,
                                            ratio = ar,
                                            hxlen = hx_list,
                                            vxlen = vx_len,
                                            echo_level = echo_code,
                                            border_clear = border_clear,
                                            calcification = calcification
                                            )
            )
            # at last we set status ok
            list_ds.append(thyroidrpc_pb2.DetectionStatus(n=nodule.n, status=0))
            t2 = time.perf_counter()
            logging.info('nodule {}, tirads time: {:.2f}'.format(nodule.n, (t2-t1)*1000))

            # save results to mongodb
            # if need update, replace old result
            if nodule.m > 0:
                old_index = nodule.m
                logging.info(f'rename nodule index: {nodule.m} -> {nodule.n}')
            else:
                old_index = nodule.n
                logging.info(f'and/replace nodule: {nodule.n}')
             
            try:
                userdb.collection.find_one_and_replace({'uid': request.uid,
                                                       'scanID': request.scanID,
                                                       'imageID': request.imageID,
                                                       'nodule': old_index
                                                       },
                                                       {'uid': request.uid,
                                                        'scanID': request.scanID,
                                                        'imageID': request.imageID,
                                                        'nodule': nodule.n,
                                                        'flag_report': False, # means not used by report!
                                                        'pos': bbox_pos,
                                                        'pos_extend': nodule.pos,
                                                        'benign': benign,
                                                        'prob': prob,
                                                        'comet': comet,
                                                        'constitute': constitute,
                                                        'shape': shape,
                                                        'echo_level': echo_code,
                                                        'border_clear': border_clear,
                                                        'cut': nodule.s,
                                                        'ratio': ar,
                                                        'hxlen': hx_list,
                                                        'vxlen': vx_len,
                                                        'calcification': calcification
                                                       },
                                                      upsert=True # if not find, just insert
                                                     )
            except:
                logging.error('userdb.collection.find_one_and_replace fail!')
                return thyroidrpc_pb2.ProtoResponse(code=DB_REPLACE_FAIL,
                                                    msg=ErrorDict[DB_REPLACE_FAIL],
                                                    data=anypb)
            
        # pack the proto result
        anypb.Pack(thyroidrpc_pb2.CTResponse(nums=len(request.nodule), ds=list_ds, nodule=list_nodule, 
                                             bp=list_benign_prob, tirads=list_tirads, modify_flag = modify_flag))
        return thyroidrpc_pb2.ProtoResponse(code=0, msg='', data=anypb)

    def DeleteNodule(self, request, context):
        """ delete nodules in one image
        """

        logging.info('---------------------------------------')
        logging.info(f'deleting nodules, uid: {request.uid}, scanID: {request.scanID}, imageID: {request.imageID}')
        logging.info(f'total requst number: {len(request.noduleID)}')

        for nindex in request.noduleID:
            res = userdb.collection.delete_one({'uid': request.uid,
                                                'scanID': request.scanID,
                                                'imageID': request.imageID,
                                                'nodule': nindex})
            if res.deleted_count != 1:
                logging.error(f'deleting nodule fail at: {nindex}')
                return thyroidrpc_pb2.ProtoResponse(code=0, msg='', data=Any())
            else:
                logging.info(f'deleting nodule success: {nindex}')
                userdb.collection.update_many({'uid': request.uid,
                                               'scanID': request.scanID,
                                               'nodule': nindex,
                                               'flag_report': True}, 
					      {'$set': {'flag_report': False} }
                                             )

                userdb.report.delete_one({'uid': request.uid,
                                          'scanID': request.scanID,
                                          'nodule': nindex})

        return thyroidrpc_pb2.ProtoResponse(code=0, msg='', data=Any())

    # merge results
    def _merge_records(self, uid, scanID, index, counts):
        """ do merge records
        """

        ctResults = userdb.collection.find({'uid': uid, 'scanID': scanID, 'nodule': index, 'flag_report': False})

        # benign or malignant
        # shape 0: 规则 1: 不规则
        # border_clear  # 边界清晰模糊   0： 模糊  1： 清晰
        # ratio  # 纵横比, 生长方式：水平位、垂直位
        # hxlen, vxlen: 大小：前后径、左右径、上下径
        # comet: 彗星尾数量， 不过不要显示数量，直接大于1时显示有彗星尾就行
        # bounding box
        benign = 0
        prob = 0
        shape = 0
        border_clear = 0
        ratio = 0
        comet = 0
        calcification = [0, 0, 0]
        xmin = 10000
        ymin = 10000
        xmax = 0
        ymax = 0

        hxlen = [0, 0]
        vxlen = 0
        pos_extend = 0

        for doc in ctResults:
            if doc['prob'] > prob:
                prob = doc['prob']
                benign = doc['benign']

            if doc['shape'] == 1:
                shape = 1

            border_clear += doc['border_clear']
            ratio = max(ratio, doc['ratio'])
            comet = max(comet, doc['comet'])
            calcification[0] = max(calcification[0], doc['calcification'][0])
            calcification[1] = max(calcification[1], doc['calcification'][1])
            calcification[2] = max(calcification[2], doc['calcification'][2])
            xmin = min(xmin, doc['pos'][0])
            ymin = min(ymin, doc['pos'][1])
            xmax = max(xmax, doc['pos'][0] + doc['pos'][2])
            ymax = max(ymax, doc['pos'][1] + doc['pos'][3])

            hxlen[0] = max(hxlen[0], doc['hxlen'][0])
            hxlen[1] = max(hxlen[1], doc['hxlen'][1])
            vxlen = max(vxlen, doc['vxlen'])
            pos_extend = doc['pos_extend']

        border_avg = border_clear/counts
        if border_avg < 0.5:
            border_clear = 0
        else:
            border_clear = 1

        # 3. constitute 构成  0： 实性    1： 实性为主  2： 囊性为主  3： 囊性
        if benign == 1:
            constitute = 2
        else:
            constitute = 1

        # 4. echo level
        # TODO: echo_level
        echo_level = 2

        # at last update records
        userdb.collection.update_many({'uid': uid, 'scanID': scanID, 'nodule': index, 'flag_report': False},
                                      {'$set': {'flag_report': True}})

        record = {'uid': uid,
                  'scanID': scanID,
                  'nodule': index,
                  'flag_modified': False,   # means the doctor does not modify the report!
                  'pos': [xmin, ymin, xmax-xmin, ymax-ymin],
                  'pos_extend': pos_extend,  # 九宫格位置
                  'benign': benign,
                  'prob': prob,
                  'shape': shape,
                  'border_clear': border_clear,
                  'constitute': constitute,
                  'echo_level': echo_level,
                  'cut': 0,  # just a placeholder, meaningless
                  'ratio': ratio,
                  'hxlen': hxlen,
                  'vxlen': vxlen,
                  'comet': comet,
                  'calcification': calcification
                  }

        return record

    def _merge_record_and_report(self, one_record, report):
        """ merge one record and report
        if report has been modified, modified result is more reliable!
        """
        org_report = report.copy()

        # TODO: merge when doctor has modified the result
        if report['flag_modified']:
            pass
        else:  # use simple merge rules!
            if one_record['prob'] > report['prob']:
                report['prob'] = one_record['prob']
                report['benign'] = one_record['benign']

            if report['shape'] == 1 or one_record['shape'] == 1:
                report['shape'] == 1

            if report['border_clear'] + one_record['border_clear'] > 1:
                report['border_clear'] = 1


            report['ratio'] = max(report['ratio'], one_record['ratio'])
            report['comet'] = max(report['comet'], one_record['comet'])

            report['calcification'][0] = max(report['calcification'][0], one_record['calcification'][0])
            report['calcification'][1] = max(report['calcification'][1], one_record['calcification'][1])
            report['calcification'][2] = max(report['calcification'][2], one_record['calcification'][2])

            report['pos'][0] = max(report['pos'][0], one_record['pos'][0])
            report['pos'][1] = max(report['pos'][1], one_record['pos'][1])

            xmax = report['pos'][0] + report['pos'][2]
            ymax = report['pos'][1] + report['pos'][3]

            xmax2 = one_record['pos'][0] + one_record['pos'][2]
            ymax2 = one_record['pos'][1] + one_record['pos'][3]

            xmax = max(xmax, xmax2)
            ymax = max(ymax, ymax2)

            report['pos'][2] = xmax - report['pos'][0]
            report['pos'][3] = ymax - report['pos'][1]

            report['hxlen'][0] = max(report['hxlen'][0], one_record['hxlen'][0])
            report['hxlen'][1] = max(report['hxlen'][1], one_record['hxlen'][1])
            report['vxlen'] = max(report['vxlen'], one_record['vxlen'])

            # TODO: echo_level
            report['echo_level'] = 2

            userdb.report.find_one_and_replace(org_report, report)


    def GenerateReport(self, request, context):
        """generating report function
        """

        logging.info('---------------------------------------')
        logging.info('GenerateReport')
        t1 = time.perf_counter()

        logging.info(f'uid: {request.uid}, scanID: {request.scanID}')

        # we should only process the un-generate report records
        # record has flag *flag_report*
        # at first count module appearing times in all images
        nodule_dict = {}  # key(nodule index):  value(appear times) 
        for user in userdb.collection.find({'uid': request.uid, 'scanID': request.scanID, 'flag_report': False}):
            key = user['nodule']
            nodule_dict[key] = nodule_dict.get(key, 0) + 1

        # merge results to one record
        # loop over nodule index
        for index, value in nodule_dict.items():
            logging.info(f'nodule index: {index}, un-gerated records: {value}')
            if value > 1:
                # merge results
                one_record = self._merge_records(request.uid, request.scanID, index, value)
            else:
                # set *flag_report* True, means this record has been used in final report!
                one_record = userdb.collection.find_one_and_update({'uid': request.uid, 'scanID': request.scanID, 'nodule': index},
                                                                   {'$set': {'flag_report': True}}, 
                                                                   return_document=pymongo.ReturnDocument.AFTER)
 
                del one_record['_id']
                del one_record['imageID']
                del one_record['flag_report']
                one_record['flag_modified'] = False   # means the doctor does not modify the report!

           # merge one_record and report
            report = userdb.report.find_one({'uid': request.uid, 'scanID': request.scanID, 'nodule': index})

            # if no report, just insert one!
            if report is None:
                userdb.report.insert_one(one_record)
            else:
                self._merge_record_and_report(one_record, report)

        # output list
        list_ds = []
        list_nodule = []
        list_benign_prob = []
        list_tirads = []

        # return mongodb report
        for nodule_report in userdb.report.find({'uid': request.uid, 'scanID': request.scanID}):

            #nodule index
            nindex = nodule_report['nodule']

            # status
            list_ds.append(thyroidrpc_pb2.DetectionStatus(n=nindex, status=0))

            # position
            x, y, w, h, = nodule_report['pos']
            s = nodule_report['cut']
            pos_extend = nodule_report['pos_extend']

            list_nodule.append(thyroidrpc_pb2.NoduleWithNum(n=nindex, m=0, x=x, y=y, w=w, h=h, s=s, pos=pos_extend))

            # classification
            benign = nodule_report['benign']
            prob = nodule_report['prob']
            list_benign_prob.append(thyroidrpc_pb2.BenignAndProb(benign=benign, prob=prob))

            # tirads
            constitute = nodule_report['constitute']
            comet = nodule_report['comet']
            shape = nodule_report['shape']
            border_clear = nodule_report['border_clear']
            echo_level = nodule_report['echo_level']
            ratio = nodule_report['ratio']
            hxlen = nodule_report['hxlen']
            vxlen = nodule_report['vxlen']
            calcification = nodule_report['calcification']

            list_tirads.append(
                thyroidrpc_pb2.OneTiradsRes(constitute = constitute, 
                                            comet = comet,
                                            shape = shape,
                                            ratio = ratio,
                                            hxlen = hxlen,
                                            vxlen = vxlen,
                                            echo_level = echo_level,
                                            border_clear = border_clear,
                                            calcification = calcification
                                            )
                              )

        nums = userdb.report.count_documents({'uid': request.uid, 'scanID': request.scanID})
        anypb = Any()
        anypb.Pack(thyroidrpc_pb2.CTResponse(nums=nums, ds=list_ds, nodule=list_nodule, 
                                             bp=list_benign_prob, tirads=list_tirads))

        t2 = time.perf_counter()
        logging.info('generate report: {:.2f}'.format((t2-t1)*1000))
        return thyroidrpc_pb2.ProtoResponse(code=0, msg='', data=anypb)


    def ModifyReport(self, request, context):
        """ modify report, only support modify one nodule per call!
        """

        """
        'benign': benign,
        'prob': prob,
        'shape': shape,
        'border_clear': border_clear,
        'constitute': constitute,
        'echo_level': echo_level,
        'cut': 0,  # just a placeholder, meaningless
        'ratio': ratio,
        'hxlen': hxlen,
        'vxlen': vxlen,
        'comet': comet,
        'calcification': calcification
        """

        logging.info('---------------------------------------')
        logging.info('modifing report')

        # 1. update benign
        if len(request.benign) == 1:
            newval = request.benign[0]
            userdb.report.find_one_and_update({'uid': request.uid, 'scanID': request.scanID, 'nodule': request.nindex},
                                              {'$set': {'flag_modified': True, 'benign': newval}})
        # 2. update prob
        if len(request.prob) == 1:
            newval = request.prob[0]
            userdb.report.find_one_and_update({'uid': request.uid,
                                               'scanID': request.scanID,
                                               'nodule': request.nindex},
                                               {'$set': {'flag_modified': True, 'prob': newval}})
        # 3. update constitute
        if len(request.constitute) == 1:
            newval = request.constitute[0]
            userdb.report.find_one_and_update({'uid': request.uid,
                                               'scanID': request.scanID,
                                               'nodule': request.nindex},
                                               {'$set': {'flag_modified': True, 'constitute': newval}})

        # 4. update comet
        if len(request.comet) == 1:
            newval = request.comet[0]
            userdb.report.find_one_and_update({'uid': request.uid,
                                               'scanID': request.scanID,
                                               'nodule': request.nindex},
                                               {'$set': {'flag_modified': True, 'comet': newval}})
        # 5. update shape
        if len(request.shape) == 1:
            newval = request.shape[0]
            userdb.report.find_one_and_update({'uid': request.uid,
                                               'scanID': request.scanID,
                                               'nodule': request.nindex},
                                               {'$set': {'flag_modified': True, 'shape': newval}})

        # 6. update echo_level
        if len(request.echo_level) == 1:
            newval = request.echo_level[0]
            doc = userdb.report.find_one({'uid': request.uid,
                                          'scanID': request.scanID,
                                          'nodule': request.nindex})
            logging.info('modify-before')
            logging.info(doc)

            userdb.report.find_one_and_update({'uid': request.uid,
                                               'scanID': request.scanID,
                                               'nodule': request.nindex},
                                               {'$set': {'flag_modified': True, 'echo_level': newval}})
            doc = userdb.report.find_one({'uid': request.uid,
                                          'scanID': request.scanID,
                                          'nodule': request.nindex})
            logging.info('modify-after')
            logging.info(doc)
        # 7. update border_clear
        if len(request.border_clear) == 1:
            newval = request.border_clear[0]
            userdb.report.find_one_and_update({'uid': request.uid,
                                               'scanID': request.scanID,
                                               'nodule': request.nindex},
                                               {'$set': {'flag_modified': True, 'border_clear': newval}})

        # 8. update calcification
        if len(request.calcification) == 3:
            newval = request.calcification
            userdb.report.find_one_and_update({'uid': request.uid,
                                               'scanID': request.scanID,
                                               'nodule': nindex},
                                               {'$set': {'flag_modified': True, 'calcification': newval}})

        return thyroidrpc_pb2.ProtoResponse(code=0, msg='', data=Any())


    def ReadReport(self, request, context):
        """ read final report
        """
        logging.info('---------------------------------------')
        logging.info('ReadReport')
        t1 = time.perf_counter()

        # output list
        list_ds = []
        list_nodule = []
        list_benign_prob = []
        list_tirads = []

        # return mongodb report
        for nodule_report in userdb.report.find({'uid': request.uid, 'scanID': request.scanID}):

            #nodule index
            nindex = nodule_report['nodule']

            # status
            list_ds.append(thyroidrpc_pb2.DetectionStatus(n=nindex, status=0))

            # position
            x, y, w, h, = nodule_report['pos']
            s = nodule_report['cut']
            pos_extend = nodule_report['pos_extend']

            list_nodule.append(thyroidrpc_pb2.NoduleWithNum(n=nindex, m=0, x=x, y=y, w=w, h=h, s=s, pos=pos_extend))

            # classification
            benign = nodule_report['benign']
            prob = nodule_report['prob']
            list_benign_prob.append(thyroidrpc_pb2.BenignAndProb(benign=benign, prob=prob))

            # tirads
            constitute = nodule_report['constitute']
            comet = nodule_report['comet']
            shape = nodule_report['shape']
            border_clear = nodule_report['border_clear']
            echo_level = nodule_report['echo_level']
            ratio = nodule_report['ratio']
            hxlen = nodule_report['hxlen']
            vxlen = nodule_report['vxlen']
            calcification = nodule_report['calcification']

            list_tirads.append(
                thyroidrpc_pb2.OneTiradsRes(constitute = constitute, 
                                            comet = comet,
                                            shape = shape,
                                            ratio = ratio,
                                            hxlen = hxlen,
                                            vxlen = vxlen,
                                            echo_level = echo_level,
                                            border_clear = border_clear,
                                            calcification = calcification
                                            )
                              )

        nums = userdb.report.count_documents({'uid': request.uid, 'scanID': request.scanID})
        anypb = Any()
        anypb.Pack(thyroidrpc_pb2.CTResponse(nums=nums, ds=list_ds, nodule=list_nodule, 
                                             bp=list_benign_prob, tirads=list_tirads))

        t2 = time.perf_counter()
        logging.info('read report: {:.2f}'.format((t2-t1)*1000))
        return thyroidrpc_pb2.ProtoResponse(code=0, msg='', data=anypb)


def serve():
    options = [('grpc.max_send_message_length', 100 * 1024 * 1024),
               ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    thyroidrpc_pb2_grpc.add_ThyroidaiGrpcServicer_to_server(ThyroidaiGrpc(), server)
    server.add_insecure_port('[::]:' + args.port)
    server.start()
    logging.info('Listen port {}!'.format(args.port))
    print(f'server is now listening port {args.port}!')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    
    # logging dir setup
    # default is /root/log, if not use current dir
    if not os.path.isdir(args.logpath):
        args.logpath = './'

    LOG_FILENAME = os.path.join(args.logpath, 'logging_server.txt')
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILENAME,
        maxBytes=5*1024*1024, # set each log file no more than 5M
        backupCount=100,
    )

    logging.basicConfig(
          format='%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d] %(levelname)s: %(message)s',
					datefmt='%Y-%m-%d %H:%M:%S', 
					level=args.level,
          handlers = [handler]
          )
    serve()
    
    
    '''
    ################################################
    ### 1. test first stage real time detection  ###
    ################################################
    demo = ThyroidaiGrpc()
    isRaw = True

    if isRaw:
        img = cv2.imread('../images/03.jpg', 0)
        h, w = img.shape
        img=img.tostring()
    else:
        with open('../images/03.jpg', 'rb') as f:
            img = f.read()
        h, w = cv2.imread('../images/03.jpg', 0).shape
    request = thyroidrpc_pb2.DetectRequest(isRaw=isRaw, image=img, height=h, width=w)
    response = demo.Detect(request,1)

    if (response.code !=0 ):
        print('error code: {}'.format(response.code))
        print('error message: {}'.format(response.msg))
    else:
        anypb = Any()
        anypb.CopyFrom(response.data)

        nodules = thyroidrpc_pb2.Nodules()
        anypb.Unpack(nodules)

        print(f'nodule number is: {nodules.nums}')
        for node in nodules.nodule:
            print(f'nudule {node.n}: ({node.x}, {node.y}, {node.w}, {node.h})')
    
    response = demo.ImageOutput(request,1)
    if (response.code !=0 ):
        print('error code: {}'.format(response.code))
        print('error message: {}'.format(response.msg))
    else:
        anypb = Any()
        anypb.CopyFrom(response.data)
        image_results = thyroidrpc_pb2.ImageRequest()
        anypb.Unpack(image_results)
    print('ImageOut')
    image = np.frombuffer(image_results.image, dtype=np.uint8)
    image = image.reshape([h, w])
    cv2.imwrite("new_image.png", image)
    '''
    
    '''
    ###########################################
    ### 2. test second stage inference      ###
    ###########################################
    print('----------------------')
    print('Second stage detect')
    print('----------------------')
    print('')

    user_id = '123456'
    scan_id = int(time.time()*10)
    image_id = 1

    list_nodule = []

    cut = 1
    for node in nodules.nodule:
        """
        # extending bounding box to 1.5
        nx = max(0, int(node.x - 0.25 * node.w))
        ny = max(0, int(node.y - 0.25 * node.h))
        nw = min(w, int(node.w * 1.5))
        nh = min(h, int(node.h * 1.5))
        """
        nx, ny, nw, nh = (node.x, node.y, node.w, node.h)
        list_nodule.append(thyroidrpc_pb2.NoduleWithNum(n=node.n, m=0, x=nx, y=ny, w=nw, h=nh, s=cut, pos=5))
        if cut == 1:
            cut = 2
        elif cut == 2:
            cut = 1
        else:
            pass

    response = demo.ClassifyAndTirads(
            thyroidrpc_pb2.CTRequest(
                uid=user_id,
                scanID=scan_id,
                imageID=image_id,
                isRaw=isRaw,
                image=img, 
                height=h, 
                width=w, 
                ppc=160, 
                nodule=list_nodule
            ), 1)

    if (response.code !=0 ):
        print('error code: {}'.format(response.code))
        print('error message: {}'.format(response.msg))
    else:
        anypb = Any()
        anypb.CopyFrom(response.data)

        ctRes = thyroidrpc_pb2.CTResponse()
        anypb.Unpack(ctRes)

        print(f'total nodule number: {ctRes.nums}')
        print()

        for ds, nodule, bp, tirads in zip(ctRes.ds, ctRes.nodule, ctRes.bp, ctRes.tirads):
            print(f'nodule index: {ds.n}:')
            if ds.status != 0:
                print('nodule {} not detect !'.format(ds.n))
            else:
                print('{:>25}: ({}, {}, {}, {})'.format('nodule x:', nodule.x, nodule.y, 
                                                                        nodule.w, nodule.h))
                print('{:>25}: {}'.format('benign', bp.benign))
                print('{:>25}: {}'.format('corresponding prob', bp.prob))
                print('{:>25}: {}'.format('constitute', tirads.constitute))
                print('{:>25}: {}'.format('Comet tail quantity', tirads.comet))
                print('{:>25}: {}'.format('Shape', tirads.shape))
                print('{:>25}: {}'.format('Aspect ratio', tirads.ratio))
                print('{:>25}: {}'.format('Horizontal axis(cm)', tirads.hxlen))
                print('{:>25}: {}'.format('Vertical axis(cm)', tirads.vxlen))
                print('{:>25}: {}'.format('echo level', tirads.echo_level))
                print('{:>25}: {}'.format('border_clear', tirads.border_clear))
                print('{:>25}: {}'.format('calcification', tirads.calcification[0]))
                print('{:>25}: {}'.format('calcification', tirads.calcification[1]))
                print('{:>25}: {}'.format('calcification', tirads.calcification[2]))
                print('{:>25}: {}'.format('calcification', tirads.calcification[3]))
                print()

    print('------------------------')
    print('Second stage re-request')
    print('------------------------')
    print('')
    list_nodule = []

    cut = 1
    for node in nodules.nodule:
        # extending bounding box to 1.5
        nx = max(0, int(node.x - 0.25 * node.w))
        ny = max(0, int(node.y - 0.25 * node.h))
        nw = min(w, int(node.w * 1.5))
        nh = min(h, int(node.h * 1.5))

        if node.n == 1:
            list_nodule.append(thyroidrpc_pb2.NoduleWithNum(n=node.n, m=0, x=nx, y=ny, w=nw, h=nh, s=cut, pos=5))
        else:
            list_nodule.append(thyroidrpc_pb2.NoduleWithNum(n=node.n+1, m=node.n, x=nx, y=ny, w=nw, h=nh, s=cut, pos=5))

        if cut == 1:
            cut = 2
        elif cut == 2:
            cut = 1
        else:
            pass

    response = demo.ClassifyAndTirads(
            thyroidrpc_pb2.CTRequest(
                uid=user_id,
                scanID=scan_id,
                imageID=image_id,
                isRaw=isRaw,
                image=img, 
                height=h, 
                width=w, 
                ppc=160, 
                nodule=list_nodule
            ),1)

    if (response.code !=0 ):
        print('error code: {}'.format(response.code))
        print('error message: {}'.format(response.msg))
    else:
        anypb = Any()
        anypb.CopyFrom(response.data)

        ctRes = thyroidrpc_pb2.CTResponse()
        anypb.Unpack(ctRes)

        print(f'total nodule number: {ctRes.nums}')
        print()

        for ds, nodule, bp, tirads in zip(ctRes.ds, ctRes.nodule, ctRes.bp, ctRes.tirads):
            print(f'nodule index: {ds.n}:')
            if ds.status != 0:
                print('nodule {} not detect !'.format(ds.n))
            else:
                print('{:>25}: ({}, {}, {}, {})'.format('nodule x:', nodule.x, nodule.y, 
                                                                        nodule.w, nodule.h))
                print('{:>25}: {}'.format('benign', bp.benign))
                print('{:>25}: {}'.format('corresponding prob', bp.prob))
                print('{:>25}: {}'.format('constitute', tirads.constitute))
                print('{:>25}: {}'.format('Comet tail quantity', tirads.comet))
                print('{:>25}: {}'.format('Shape', tirads.shape))
                print('{:>25}: {}'.format('Aspect ratio', tirads.ratio))
                print('{:>25}: {}'.format('Horizontal axis(cm)', tirads.hxlen))
                print('{:>25}: {}'.format('Vertical axis(cm)', tirads.vxlen))
                print('{:>25}: {}'.format('echo level', tirads.echo_level))
                print('{:>25}: {}'.format('border_clear', tirads.border_clear))
                print('{:>25}: {}'.format('calcification', tirads.calcification[0]))
                print('{:>25}: {}'.format('calcification', tirads.calcification[1]))
                print('{:>25}: {}'.format('calcification', tirads.calcification[2]))
                print('{:>25}: {}'.format('calcification', tirads.calcification[3]))
                print()

    print('----------------------')
    print('deleting nodule')
    print('----------------------')
    print('')
    '''
