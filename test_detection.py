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


if __name__ == '__main__':
    logging.basicConfig()
    run()

