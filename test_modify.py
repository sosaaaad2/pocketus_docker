"""The Python implementation of the GRPC thyroid ai client."""

import argparse
import logging

import grpc
import thyroidrpc_pb2
import thyroidrpc_pb2_grpc
from google.protobuf.any_pb2 import Any

import cv2
import numpy as np
import time


parser = argparse.ArgumentParser(description='thyroid ai client.')

parser.add_argument(
    '-s',
    '--server',
    type=str,
    default='localhost',
    required=False,
    help="set server ip, default is localhost"
)

parser.add_argument(
    '-p',
    '--port',
    type=str,
    default='50051',
    required=False,
    help="set server port, default is 50052"
)

args = parser.parse_args()

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    if (':' in args.server):
        ip = args.server
    else:
        ip = args.server + ':' + args.port
    print(f'connecting server: {ip}')

    with grpc.insecure_channel(ip) as channel:
        stub = thyroidrpc_pb2_grpc.ThyroidaiGrpcStub(channel)

        ################################################
        ### 1. test first stage real time detection  ###
        ################################################
        isRaw = True

        if isRaw:
            img = cv2.imread('./images/190528152340059.png', 0)
            h, w = img.shape
            img=img.tostring()
        else:
            with open('./images/190528152340059.png', 'rb') as f:
                img = f.read()
            h, w = cv2.imread('./images/190528152340059.png', 0).shape


        response = stub.Detect(thyroidrpc_pb2.DetectRequest(isRaw=isRaw, image=img, height=h, width=w))
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

        response = stub.ClassifyAndTirads(
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
                ))

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
                    print()

        response = stub.GenerateReport(thyroidrpc_pb2.UserID(uid=user_id, scanID=scan_id))
        if (response.code !=0 ):
            print('error code: {}'.format(response.code))

        print('----------------------')
        print('modify report')
        print('----------------------')
        print('')

        response = stub.ModifyReport(thyroidrpc_pb2.ModifyCT(uid=user_id, scanID=scan_id, nindex=1, 
                                                             benign=[2],
                                                             prob=[0.8],
                                                             constitute=[2],
                                                             shape=[0],
                                                             echo_level=[3],
                                                             border_clear=[0],
                                                             calcification=[1,0,0]
                                                             ))
        if (response.code !=0 ):
            print('error code: {}'.format(response.code))


        response = stub.ReadReport(thyroidrpc_pb2.UserID(uid=user_id, scanID=scan_id))
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
                    print('{:>25}: {}'.format('frond-end (cm)', tirads.hxlen[0]))
                    print('{:>25}: {}'.format('left-right(cm)', tirads.hxlen[1]))
                    print('{:>25}: {}'.format('up-down (cm)', tirads.vxlen))
                    print('{:>25}: {}'.format('pos_extend', nodule.pos))
                    print('{:>25}: {}'.format('echo level', tirads.echo_level))
                    print('{:>25}: {}'.format('border_clear', tirads.border_clear))
                    print('{:>25}: {}'.format('calcification', tirads.calcification[0]))
                    print('{:>25}: {}'.format('calcification', tirads.calcification[1]))
                    print('{:>25}: {}'.format('calcification', tirads.calcification[2]))
                    print()


if __name__ == '__main__':
    logging.basicConfig()
    run()

