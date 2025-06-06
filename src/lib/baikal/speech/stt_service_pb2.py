# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: baikal/speech/stt_service.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from baikal.speech import recognition_config_pb2 as baikal_dot_speech_dot_recognition__config__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1f\x62\x61ikal/speech/stt_service.proto\x12\rbaikal.speech\x1a&baikal/speech/recognition_config.proto\"R\n\x10RecognizeRequest\x12\x0c\n\x04wave\x18\x01 \x01(\x0c\x12\x30\n\x06\x63onfig\x18\x02 \x01(\x0b\x32 .baikal.speech.RecognitionConfig\"w\n\x16StreamRecognizeRequest\x12\x38\n\x06\x63onfig\x18\x01 \x01(\x0b\x32&.baikal.speech.StreamRecognitionConfigH\x00\x12\x0e\n\x04wave\x18\x02 \x01(\x0cH\x00\x42\x13\n\x11streaming_request\"\xa2\x02\n\x11RecognizeResponse\x12\x12\n\ntranscript\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x01\x12\x38\n\x05words\x18\x03 \x03(\x0b\x32).baikal.speech.RecognizeResponse.TextSpan\x12\x38\n\x05\x63hars\x18\x04 \x03(\x0b\x32).baikal.speech.RecognizeResponse.TextSpan\x12;\n\x08phonemes\x18\x05 \x03(\x0b\x32).baikal.speech.RecognizeResponse.TextSpan\x1a\x34\n\x08TextSpan\x12\r\n\x05start\x18\x01 \x01(\x02\x12\x0b\n\x03\x65nd\x18\x02 \x01(\x02\x12\x0c\n\x04text\x18\x03 \x01(\t2\xa4\x02\n\x10RecognizeService\x12P\n\tRecognize\x12\x1f.baikal.speech.RecognizeRequest\x1a .baikal.speech.RecognizeResponse\"\x00\x12\\\n\rLongRecognize\x12%.baikal.speech.StreamRecognizeRequest\x1a .baikal.speech.RecognizeResponse\"\x00(\x01\x12`\n\x0fStreamRecognize\x12%.baikal.speech.StreamRecognizeRequest\x1a .baikal.speech.RecognizeResponse\"\x00(\x01\x30\x01\x42\x38\x42\x15RecognizeServiceProtoP\x01Z\x1d\x62\x61ikal.ai/proto/baikal/speechb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'baikal.speech.stt_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'B\025RecognizeServiceProtoP\001Z\035baikal.ai/proto/baikal/speech'
  _globals['_RECOGNIZEREQUEST']._serialized_start=90
  _globals['_RECOGNIZEREQUEST']._serialized_end=172
  _globals['_STREAMRECOGNIZEREQUEST']._serialized_start=174
  _globals['_STREAMRECOGNIZEREQUEST']._serialized_end=293
  _globals['_RECOGNIZERESPONSE']._serialized_start=296
  _globals['_RECOGNIZERESPONSE']._serialized_end=586
  _globals['_RECOGNIZERESPONSE_TEXTSPAN']._serialized_start=534
  _globals['_RECOGNIZERESPONSE_TEXTSPAN']._serialized_end=586
  _globals['_RECOGNIZESERVICE']._serialized_start=589
  _globals['_RECOGNIZESERVICE']._serialized_end=881
# @@protoc_insertion_point(module_scope)
