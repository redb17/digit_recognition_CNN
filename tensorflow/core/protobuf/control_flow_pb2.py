# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/control_flow.proto

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
  name='tensorflow/core/protobuf/control_flow.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_pb=_b('\n+tensorflow/core/protobuf/control_flow.proto\x12\ntensorflow\"\x96\x01\n\tValuesDef\x12\x0e\n\x06values\x18\x01 \x03(\t\x12\x42\n\x0f\x65xternal_values\x18\x02 \x03(\x0b\x32).tensorflow.ValuesDef.ExternalValuesEntry\x1a\x35\n\x13\x45xternalValuesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x83\x01\n\x15\x43ontrolFlowContextDef\x12/\n\tcond_ctxt\x18\x01 \x01(\x0b\x32\x1a.tensorflow.CondContextDefH\x00\x12\x31\n\nwhile_ctxt\x18\x02 \x01(\x0b\x32\x1b.tensorflow.WhileContextDefH\x00\x42\x06\n\x04\x63txt\"\xc4\x01\n\x0e\x43ondContextDef\x12\x14\n\x0c\x63ontext_name\x18\x01 \x01(\t\x12\x11\n\tpred_name\x18\x02 \x01(\t\x12\x12\n\npivot_name\x18\x03 \x01(\t\x12\x0e\n\x06\x62ranch\x18\x04 \x01(\x05\x12)\n\nvalues_def\x18\x05 \x01(\x0b\x32\x15.tensorflow.ValuesDef\x12:\n\x0fnested_contexts\x18\x06 \x03(\x0b\x32!.tensorflow.ControlFlowContextDef\"\xf5\x02\n\x0fWhileContextDef\x12\x14\n\x0c\x63ontext_name\x18\x01 \x01(\t\x12\x1b\n\x13parallel_iterations\x18\x02 \x01(\x05\x12\x11\n\tback_prop\x18\x03 \x01(\x08\x12\x13\n\x0bswap_memory\x18\x04 \x01(\x08\x12\x12\n\npivot_name\x18\x05 \x01(\t\x12\x1b\n\x13pivot_for_pred_name\x18\x06 \x01(\t\x12\x1b\n\x13pivot_for_body_name\x18\x07 \x01(\t\x12\x17\n\x0floop_exit_names\x18\x08 \x03(\t\x12\x18\n\x10loop_enter_names\x18\n \x03(\t\x12)\n\nvalues_def\x18\t \x01(\x0b\x32\x15.tensorflow.ValuesDef\x12\x1f\n\x17maximum_iterations_name\x18\x0b \x01(\t\x12:\n\x0fnested_contexts\x18\x0c \x03(\x0b\x32!.tensorflow.ControlFlowContextDefBp\n\x18org.tensorflow.frameworkB\x11\x43ontrolFlowProtosP\x01Z<github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf\xf8\x01\x01\x62\x06proto3')
)




_VALUESDEF_EXTERNALVALUESENTRY = _descriptor.Descriptor(
  name='ExternalValuesEntry',
  full_name='tensorflow.ValuesDef.ExternalValuesEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.ValuesDef.ExternalValuesEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.ValuesDef.ExternalValuesEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
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
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=157,
  serialized_end=210,
)

_VALUESDEF = _descriptor.Descriptor(
  name='ValuesDef',
  full_name='tensorflow.ValuesDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='values', full_name='tensorflow.ValuesDef.values', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='external_values', full_name='tensorflow.ValuesDef.external_values', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_VALUESDEF_EXTERNALVALUESENTRY, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=60,
  serialized_end=210,
)


_CONTROLFLOWCONTEXTDEF = _descriptor.Descriptor(
  name='ControlFlowContextDef',
  full_name='tensorflow.ControlFlowContextDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cond_ctxt', full_name='tensorflow.ControlFlowContextDef.cond_ctxt', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='while_ctxt', full_name='tensorflow.ControlFlowContextDef.while_ctxt', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
    _descriptor.OneofDescriptor(
      name='ctxt', full_name='tensorflow.ControlFlowContextDef.ctxt',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=213,
  serialized_end=344,
)


_CONDCONTEXTDEF = _descriptor.Descriptor(
  name='CondContextDef',
  full_name='tensorflow.CondContextDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context_name', full_name='tensorflow.CondContextDef.context_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pred_name', full_name='tensorflow.CondContextDef.pred_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pivot_name', full_name='tensorflow.CondContextDef.pivot_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='branch', full_name='tensorflow.CondContextDef.branch', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='values_def', full_name='tensorflow.CondContextDef.values_def', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='nested_contexts', full_name='tensorflow.CondContextDef.nested_contexts', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=347,
  serialized_end=543,
)


_WHILECONTEXTDEF = _descriptor.Descriptor(
  name='WhileContextDef',
  full_name='tensorflow.WhileContextDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context_name', full_name='tensorflow.WhileContextDef.context_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='parallel_iterations', full_name='tensorflow.WhileContextDef.parallel_iterations', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='back_prop', full_name='tensorflow.WhileContextDef.back_prop', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='swap_memory', full_name='tensorflow.WhileContextDef.swap_memory', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pivot_name', full_name='tensorflow.WhileContextDef.pivot_name', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pivot_for_pred_name', full_name='tensorflow.WhileContextDef.pivot_for_pred_name', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pivot_for_body_name', full_name='tensorflow.WhileContextDef.pivot_for_body_name', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='loop_exit_names', full_name='tensorflow.WhileContextDef.loop_exit_names', index=7,
      number=8, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='loop_enter_names', full_name='tensorflow.WhileContextDef.loop_enter_names', index=8,
      number=10, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='values_def', full_name='tensorflow.WhileContextDef.values_def', index=9,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='maximum_iterations_name', full_name='tensorflow.WhileContextDef.maximum_iterations_name', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='nested_contexts', full_name='tensorflow.WhileContextDef.nested_contexts', index=11,
      number=12, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=546,
  serialized_end=919,
)

_VALUESDEF_EXTERNALVALUESENTRY.containing_type = _VALUESDEF
_VALUESDEF.fields_by_name['external_values'].message_type = _VALUESDEF_EXTERNALVALUESENTRY
_CONTROLFLOWCONTEXTDEF.fields_by_name['cond_ctxt'].message_type = _CONDCONTEXTDEF
_CONTROLFLOWCONTEXTDEF.fields_by_name['while_ctxt'].message_type = _WHILECONTEXTDEF
_CONTROLFLOWCONTEXTDEF.oneofs_by_name['ctxt'].fields.append(
  _CONTROLFLOWCONTEXTDEF.fields_by_name['cond_ctxt'])
_CONTROLFLOWCONTEXTDEF.fields_by_name['cond_ctxt'].containing_oneof = _CONTROLFLOWCONTEXTDEF.oneofs_by_name['ctxt']
_CONTROLFLOWCONTEXTDEF.oneofs_by_name['ctxt'].fields.append(
  _CONTROLFLOWCONTEXTDEF.fields_by_name['while_ctxt'])
_CONTROLFLOWCONTEXTDEF.fields_by_name['while_ctxt'].containing_oneof = _CONTROLFLOWCONTEXTDEF.oneofs_by_name['ctxt']
_CONDCONTEXTDEF.fields_by_name['values_def'].message_type = _VALUESDEF
_CONDCONTEXTDEF.fields_by_name['nested_contexts'].message_type = _CONTROLFLOWCONTEXTDEF
_WHILECONTEXTDEF.fields_by_name['values_def'].message_type = _VALUESDEF
_WHILECONTEXTDEF.fields_by_name['nested_contexts'].message_type = _CONTROLFLOWCONTEXTDEF
DESCRIPTOR.message_types_by_name['ValuesDef'] = _VALUESDEF
DESCRIPTOR.message_types_by_name['ControlFlowContextDef'] = _CONTROLFLOWCONTEXTDEF
DESCRIPTOR.message_types_by_name['CondContextDef'] = _CONDCONTEXTDEF
DESCRIPTOR.message_types_by_name['WhileContextDef'] = _WHILECONTEXTDEF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ValuesDef = _reflection.GeneratedProtocolMessageType('ValuesDef', (_message.Message,), dict(

  ExternalValuesEntry = _reflection.GeneratedProtocolMessageType('ExternalValuesEntry', (_message.Message,), dict(
    DESCRIPTOR = _VALUESDEF_EXTERNALVALUESENTRY,
    __module__ = 'tensorflow.core.protobuf.control_flow_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.ValuesDef.ExternalValuesEntry)
    ))
  ,
  DESCRIPTOR = _VALUESDEF,
  __module__ = 'tensorflow.core.protobuf.control_flow_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.ValuesDef)
  ))
_sym_db.RegisterMessage(ValuesDef)
_sym_db.RegisterMessage(ValuesDef.ExternalValuesEntry)

ControlFlowContextDef = _reflection.GeneratedProtocolMessageType('ControlFlowContextDef', (_message.Message,), dict(
  DESCRIPTOR = _CONTROLFLOWCONTEXTDEF,
  __module__ = 'tensorflow.core.protobuf.control_flow_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.ControlFlowContextDef)
  ))
_sym_db.RegisterMessage(ControlFlowContextDef)

CondContextDef = _reflection.GeneratedProtocolMessageType('CondContextDef', (_message.Message,), dict(
  DESCRIPTOR = _CONDCONTEXTDEF,
  __module__ = 'tensorflow.core.protobuf.control_flow_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.CondContextDef)
  ))
_sym_db.RegisterMessage(CondContextDef)

WhileContextDef = _reflection.GeneratedProtocolMessageType('WhileContextDef', (_message.Message,), dict(
  DESCRIPTOR = _WHILECONTEXTDEF,
  __module__ = 'tensorflow.core.protobuf.control_flow_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.WhileContextDef)
  ))
_sym_db.RegisterMessage(WhileContextDef)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\030org.tensorflow.frameworkB\021ControlFlowProtosP\001Z<github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf\370\001\001'))
_VALUESDEF_EXTERNALVALUESENTRY.has_options = True
_VALUESDEF_EXTERNALVALUESENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
# @@protoc_insertion_point(module_scope)
