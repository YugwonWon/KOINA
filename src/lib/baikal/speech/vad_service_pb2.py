# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: baikal/speech/vad_service.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1f\x62\x61ikal/speech/vad_service.proto\x12\rbaikal.speech\"\x1a\n\nVADRequest\x12\x0c\n\x04wave\x18\x01 \x01(\x0c\"C\n\x12GetVADSpanResponse\x12-\n\x0bvoice_spans\x18\x01 \x03(\x0b\x32\x18.baikal.speech.VoiseSpan\"\'\n\tVoiseSpan\x12\r\n\x05start\x18\x01 \x01(\x02\x12\x0b\n\x03\x65nd\x18\x02 \x01(\x02\x32Z\n\nVADService\x12L\n\nGetVADSpan\x12\x19.baikal.speech.VADRequest\x1a!.baikal.speech.GetVADSpanResponse\"\x00\x42\x32\x42\x0fVADServiceProtoP\x01Z\x1d\x62\x61ikal.ai/proto/baikal/speechb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'baikal.speech.vad_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'B\017VADServiceProtoP\001Z\035baikal.ai/proto/baikal/speech'
  _globals['_VADREQUEST']._serialized_start=50
  _globals['_VADREQUEST']._serialized_end=76
  _globals['_GETVADSPANRESPONSE']._serialized_start=78
  _globals['_GETVADSPANRESPONSE']._serialized_end=145
  _globals['_VOISESPAN']._serialized_start=147
  _globals['_VOISESPAN']._serialized_end=186
  _globals['_VADSERVICE']._serialized_start=188
  _globals['_VADSERVICE']._serialized_end=278
# @@protoc_insertion_point(module_scope)
