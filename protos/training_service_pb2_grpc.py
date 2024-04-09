# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import training_service_pb2 as training__service__pb2


class TrainingServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetModelWeights = channel.unary_unary(
                '/distributed_training.TrainingService/GetModelWeights',
                request_serializer=training__service__pb2.Worker.SerializeToString,
                response_deserializer=training__service__pb2.ModelWeights.FromString,
                )
        self.SendModelWeights = channel.unary_unary(
                '/distributed_training.TrainingService/SendModelWeights',
                request_serializer=training__service__pb2.ModelWeights.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.GetDataSet = channel.unary_unary(
                '/distributed_training.TrainingService/GetDataSet',
                request_serializer=training__service__pb2.DataSet.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.GetTrainingConfig = channel.unary_unary(
                '/distributed_training.TrainingService/GetTrainingConfig',
                request_serializer=training__service__pb2.TrainingConfig.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.SendGradients = channel.unary_unary(
                '/distributed_training.TrainingService/SendGradients',
                request_serializer=training__service__pb2.ModelGradients.SerializeToString,
                response_deserializer=training__service__pb2.ModelGradients.FromString,
                )


class TrainingServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetModelWeights(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendModelWeights(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDataSet(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTrainingConfig(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendGradients(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TrainingServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetModelWeights': grpc.unary_unary_rpc_method_handler(
                    servicer.GetModelWeights,
                    request_deserializer=training__service__pb2.Worker.FromString,
                    response_serializer=training__service__pb2.ModelWeights.SerializeToString,
            ),
            'SendModelWeights': grpc.unary_unary_rpc_method_handler(
                    servicer.SendModelWeights,
                    request_deserializer=training__service__pb2.ModelWeights.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'GetDataSet': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDataSet,
                    request_deserializer=training__service__pb2.DataSet.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'GetTrainingConfig': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTrainingConfig,
                    request_deserializer=training__service__pb2.TrainingConfig.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'SendGradients': grpc.unary_unary_rpc_method_handler(
                    servicer.SendGradients,
                    request_deserializer=training__service__pb2.ModelGradients.FromString,
                    response_serializer=training__service__pb2.ModelGradients.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'distributed_training.TrainingService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class TrainingService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetModelWeights(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/distributed_training.TrainingService/GetModelWeights',
            training__service__pb2.Worker.SerializeToString,
            training__service__pb2.ModelWeights.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendModelWeights(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/distributed_training.TrainingService/SendModelWeights',
            training__service__pb2.ModelWeights.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDataSet(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/distributed_training.TrainingService/GetDataSet',
            training__service__pb2.DataSet.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTrainingConfig(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/distributed_training.TrainingService/GetTrainingConfig',
            training__service__pb2.TrainingConfig.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendGradients(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/distributed_training.TrainingService/SendGradients',
            training__service__pb2.ModelGradients.SerializeToString,
            training__service__pb2.ModelGradients.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
