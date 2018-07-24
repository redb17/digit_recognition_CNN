# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/critical_section.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/protobuf/critical_section.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_pb=_b('\n/tensorflow/core/protobuf/critical_section.proto\x12\ntensorflow\"3\n\x12\x43riticalSectionDef\x12\x1d\n\x15\x63ritical_section_name\x18\x01 \x01(\t\"j\n\x1b\x43riticalSectionExecutionDef\x12(\n execute_in_critical_section_name\x18\x01 \x01(\t\x12!\n\x19\x65xclusive_resource_access\x18\x02 \x01(\x08\x42t\n\x18org.tensorflow.frameworkB\x15\x43riticalSectionProtosP\x01Z<github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf\xf8\x01\x01\x62\x06proto3')
)




_CRITICALSECTIONDEF = _descriptor.Descriptor(
  name='CriticalSectionDef',
  full_name='tensorflow.CriticalSectionDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='critical_section_name', full_name='tensorflow.CriticalSectionDef.critical_section_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=63,
  serialized_end=114,
)


_CRITICALSECTIONEXECUTIONDEF = _descriptor.Descriptor(
  name='CriticalSectionExecutionDef',
  full_name='tensorflow.CriticalSectionExecutionDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='execute_in_critical_section_name', full_name='tensorflow.CriticalSectionExecutionDef.execute_in_critical_section_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='exclusive_resource_access', full_name='tensorflow.CriticalSectionExecutionDef.exclusive_resource_access', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=116,
  serialized_end=222,
)

DESCRIPTOR.message_types_by_name['CriticalSectionDef'] = _CRITICALSECTIONDEF
DESCRIPTOR.message_types_by_name['CriticalSectionExecutionDef'] = _CRITICALSECTIONEXECUTIONDEF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CriticalSectionDef = _reflection.GeneratedProtocolMessageType('CriticalSectionDef', (_message.Message,), dict(
  DESCRIPTOR = _CRITICALSECTIONDEF,
  __module__ = 'tensorflow.core.protobuf.critical_section_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.CriticalSectionDef)
  ))
_sym_db.RegisterMessage(CriticalSectionDef)

CriticalSectionExecutionDef = _reflection.GeneratedProtocolMessageType('CriticalSectionExecutionDef', (_message.Message,), dict(
  DESCRIPTOR = _CRITICALSECTIONEXECUTIONDEF,
  __module__ = 'tensorflow.core.protobuf.critical_section_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.CriticalSectionExecutionDef)
  ))
_sym_db.RegisterMessage(CriticalSectionExecutionDef)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\030org.tensorflow.frameworkB\025CriticalSectionProtosP\001Z<github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf\370\001\001'))
# @@protoc_insertion_point(module_scope)
