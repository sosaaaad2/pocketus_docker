# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import thyroidrpc_pb2 as thyroidrpc__pb2


class ThyroidaiGrpcStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ImageOutput = channel.unary_unary(
                '/thyroidaiproto.ThyroidaiGrpc/ImageOutput',
                request_serializer=thyroidrpc__pb2.DetectRequestNew.SerializeToString,
                response_deserializer=thyroidrpc__pb2.ProtoResponse.FromString,
                )
        self.Detect = channel.unary_unary(
                '/thyroidaiproto.ThyroidaiGrpc/Detect',
                request_serializer=thyroidrpc__pb2.DetectRequest.SerializeToString,
                response_deserializer=thyroidrpc__pb2.ProtoResponse.FromString,
                )
        self.ClassifyAndTirads = channel.unary_unary(
                '/thyroidaiproto.ThyroidaiGrpc/ClassifyAndTirads',
                request_serializer=thyroidrpc__pb2.CTRequest.SerializeToString,
                response_deserializer=thyroidrpc__pb2.ProtoResponse.FromString,
                )
        self.DeleteNodule = channel.unary_unary(
                '/thyroidaiproto.ThyroidaiGrpc/DeleteNodule',
                request_serializer=thyroidrpc__pb2.DeleteNoduleRequest.SerializeToString,
                response_deserializer=thyroidrpc__pb2.ProtoResponse.FromString,
                )
        self.GenerateReport = channel.unary_unary(
                '/thyroidaiproto.ThyroidaiGrpc/GenerateReport',
                request_serializer=thyroidrpc__pb2.UserID.SerializeToString,
                response_deserializer=thyroidrpc__pb2.ProtoResponse.FromString,
                )
        self.ModifyReport = channel.unary_unary(
                '/thyroidaiproto.ThyroidaiGrpc/ModifyReport',
                request_serializer=thyroidrpc__pb2.ModifyCT.SerializeToString,
                response_deserializer=thyroidrpc__pb2.ProtoResponse.FromString,
                )
        self.ReadReport = channel.unary_unary(
                '/thyroidaiproto.ThyroidaiGrpc/ReadReport',
                request_serializer=thyroidrpc__pb2.UserID.SerializeToString,
                response_deserializer=thyroidrpc__pb2.ProtoResponse.FromString,
                )


class ThyroidaiGrpcServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ImageOutput(self, request, context):
        """! image output
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Detect(self, request, context):
        """! detection task
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ClassifyAndTirads(self, request, context):
        """! classification and tirads task
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteNodule(self, request, context):
        """! delete nodule in the image
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GenerateReport(self, request, context):
        """! generate report
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ModifyReport(self, request, context):
        """! modify report
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ReadReport(self, request, context):
        """! read report
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ThyroidaiGrpcServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ImageOutput': grpc.unary_unary_rpc_method_handler(
                    servicer.ImageOutput,
                    request_deserializer=thyroidrpc__pb2.DetectRequestNew.FromString,
                    response_serializer=thyroidrpc__pb2.ProtoResponse.SerializeToString,
            ),
            'Detect': grpc.unary_unary_rpc_method_handler(
                    servicer.Detect,
                    request_deserializer=thyroidrpc__pb2.DetectRequest.FromString,
                    response_serializer=thyroidrpc__pb2.ProtoResponse.SerializeToString,
            ),
            'ClassifyAndTirads': grpc.unary_unary_rpc_method_handler(
                    servicer.ClassifyAndTirads,
                    request_deserializer=thyroidrpc__pb2.CTRequest.FromString,
                    response_serializer=thyroidrpc__pb2.ProtoResponse.SerializeToString,
            ),
            'DeleteNodule': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteNodule,
                    request_deserializer=thyroidrpc__pb2.DeleteNoduleRequest.FromString,
                    response_serializer=thyroidrpc__pb2.ProtoResponse.SerializeToString,
            ),
            'GenerateReport': grpc.unary_unary_rpc_method_handler(
                    servicer.GenerateReport,
                    request_deserializer=thyroidrpc__pb2.UserID.FromString,
                    response_serializer=thyroidrpc__pb2.ProtoResponse.SerializeToString,
            ),
            'ModifyReport': grpc.unary_unary_rpc_method_handler(
                    servicer.ModifyReport,
                    request_deserializer=thyroidrpc__pb2.ModifyCT.FromString,
                    response_serializer=thyroidrpc__pb2.ProtoResponse.SerializeToString,
            ),
            'ReadReport': grpc.unary_unary_rpc_method_handler(
                    servicer.ReadReport,
                    request_deserializer=thyroidrpc__pb2.UserID.FromString,
                    response_serializer=thyroidrpc__pb2.ProtoResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'thyroidaiproto.ThyroidaiGrpc', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ThyroidaiGrpc(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ImageOutput(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thyroidaiproto.ThyroidaiGrpc/ImageOutput',
            thyroidrpc__pb2.DetectRequestNew.SerializeToString,
            thyroidrpc__pb2.ProtoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Detect(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thyroidaiproto.ThyroidaiGrpc/Detect',
            thyroidrpc__pb2.DetectRequest.SerializeToString,
            thyroidrpc__pb2.ProtoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ClassifyAndTirads(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thyroidaiproto.ThyroidaiGrpc/ClassifyAndTirads',
            thyroidrpc__pb2.CTRequest.SerializeToString,
            thyroidrpc__pb2.ProtoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteNodule(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thyroidaiproto.ThyroidaiGrpc/DeleteNodule',
            thyroidrpc__pb2.DeleteNoduleRequest.SerializeToString,
            thyroidrpc__pb2.ProtoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GenerateReport(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thyroidaiproto.ThyroidaiGrpc/GenerateReport',
            thyroidrpc__pb2.UserID.SerializeToString,
            thyroidrpc__pb2.ProtoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ModifyReport(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thyroidaiproto.ThyroidaiGrpc/ModifyReport',
            thyroidrpc__pb2.ModifyCT.SerializeToString,
            thyroidrpc__pb2.ProtoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReadReport(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thyroidaiproto.ThyroidaiGrpc/ReadReport',
            thyroidrpc__pb2.UserID.SerializeToString,
            thyroidrpc__pb2.ProtoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
