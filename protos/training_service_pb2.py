# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: training_service.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16training_service.proto\x12\x14\x64istributed_training\x1a\x1bgoogle/protobuf/empty.proto\"%\n\x06Worker\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05speed\x18\x02 \x01(\x05\",\n\x0cModelWeights\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06values\x18\x02 \x01(\x0c\".\n\x0eModelGradients\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06values\x18\x02 \x01(\x0c\"/\n\x07\x44\x61taSet\x12\x12\n\ndata_start\x18\x01 \x01(\x05\x12\x10\n\x08\x64\x61ta_end\x18\x02 \x01(\x05\"9\n\x0eTrainingConfig\x12\x15\n\rlearning_rate\x18\x01 \x01(\x02\x12\x10\n\x08momentum\x18\x02 \x01(\x02\x32\xb5\x03\n\x0fTrainingService\x12U\n\x0fGetModelWeights\x12\x1c.distributed_training.Worker\x1a\".distributed_training.ModelWeights\"\x00\x12P\n\x10SendModelWeights\x12\".distributed_training.ModelWeights\x1a\x16.google.protobuf.Empty\"\x00\x12\x45\n\nGetDataSet\x12\x1d.distributed_training.DataSet\x1a\x16.google.protobuf.Empty\"\x00\x12S\n\x11GetTrainingConfig\x12$.distributed_training.TrainingConfig\x1a\x16.google.protobuf.Empty\"\x00\x12]\n\rSendGradients\x12$.distributed_training.ModelGradients\x1a$.distributed_training.ModelGradients\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'training_service_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_WORKER']._serialized_start=77
  _globals['_WORKER']._serialized_end=114
  _globals['_MODELWEIGHTS']._serialized_start=116
  _globals['_MODELWEIGHTS']._serialized_end=160
  _globals['_MODELGRADIENTS']._serialized_start=162
  _globals['_MODELGRADIENTS']._serialized_end=208
  _globals['_DATASET']._serialized_start=210
  _globals['_DATASET']._serialized_end=257
  _globals['_TRAININGCONFIG']._serialized_start=259
  _globals['_TRAININGCONFIG']._serialized_end=316
  _globals['_TRAININGSERVICE']._serialized_start=319
  _globals['_TRAININGSERVICE']._serialized_end=756
# @@protoc_insertion_point(module_scope)