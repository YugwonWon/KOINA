"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import abc
import baikal.speech.stt_service_v1_pb2
import collections.abc
import grpc
import grpc.aio
import typing

_T = typing.TypeVar("_T")

class _MaybeAsyncIterator(collections.abc.AsyncIterator[_T], collections.abc.Iterator[_T], metaclass=abc.ABCMeta): ...

class _ServicerContext(grpc.ServicerContext, grpc.aio.ServicerContext):  # type: ignore[misc, type-arg]
    ...

class STTServiceStub:
    def __init__(self, channel: typing.Union[grpc.Channel, grpc.aio.Channel]) -> None: ...
    STT: grpc.UnaryUnaryMultiCallable[
        baikal.speech.stt_service_v1_pb2.STTRequest,
        baikal.speech.stt_service_v1_pb2.STTResponse,
    ]
    """음성 인식"""

class STTServiceAsyncStub:
    STT: grpc.aio.UnaryUnaryMultiCallable[
        baikal.speech.stt_service_v1_pb2.STTRequest,
        baikal.speech.stt_service_v1_pb2.STTResponse,
    ]
    """음성 인식"""

class STTServiceServicer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def STT(
        self,
        request: baikal.speech.stt_service_v1_pb2.STTRequest,
        context: _ServicerContext,
    ) -> typing.Union[baikal.speech.stt_service_v1_pb2.STTResponse, collections.abc.Awaitable[baikal.speech.stt_service_v1_pb2.STTResponse]]:
        """음성 인식"""

def add_STTServiceServicer_to_server(servicer: STTServiceServicer, server: typing.Union[grpc.Server, grpc.aio.Server]) -> None: ...
