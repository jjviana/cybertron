// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        (unknown)
// source: textencoding/v1/textencoding.proto

package textencodingv1

import (
	_ "google.golang.org/genproto/googleapis/api/annotations"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type EncodingRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Input           string `protobuf:"bytes,1,opt,name=input,proto3" json:"input,omitempty"`
	PoolingStrategy int32  `protobuf:"varint,2,opt,name=pooling_strategy,json=poolingStrategy,proto3" json:"pooling_strategy,omitempty"`
}

func (x *EncodingRequest) Reset() {
	*x = EncodingRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_textencoding_v1_textencoding_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *EncodingRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*EncodingRequest) ProtoMessage() {}

func (x *EncodingRequest) ProtoReflect() protoreflect.Message {
	mi := &file_textencoding_v1_textencoding_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use EncodingRequest.ProtoReflect.Descriptor instead.
func (*EncodingRequest) Descriptor() ([]byte, []int) {
	return file_textencoding_v1_textencoding_proto_rawDescGZIP(), []int{0}
}

func (x *EncodingRequest) GetInput() string {
	if x != nil {
		return x.Input
	}
	return ""
}

func (x *EncodingRequest) GetPoolingStrategy() int32 {
	if x != nil {
		return x.PoolingStrategy
	}
	return 0
}

type EncodingResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Vector []float32 `protobuf:"fixed32,1,rep,packed,name=vector,proto3" json:"vector,omitempty"`
}

func (x *EncodingResponse) Reset() {
	*x = EncodingResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_textencoding_v1_textencoding_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *EncodingResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*EncodingResponse) ProtoMessage() {}

func (x *EncodingResponse) ProtoReflect() protoreflect.Message {
	mi := &file_textencoding_v1_textencoding_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use EncodingResponse.ProtoReflect.Descriptor instead.
func (*EncodingResponse) Descriptor() ([]byte, []int) {
	return file_textencoding_v1_textencoding_proto_rawDescGZIP(), []int{1}
}

func (x *EncodingResponse) GetVector() []float32 {
	if x != nil {
		return x.Vector
	}
	return nil
}

var File_textencoding_v1_textencoding_proto protoreflect.FileDescriptor

var file_textencoding_v1_textencoding_proto_rawDesc = []byte{
	0x0a, 0x22, 0x74, 0x65, 0x78, 0x74, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x69, 0x6e, 0x67, 0x2f, 0x76,
	0x31, 0x2f, 0x74, 0x65, 0x78, 0x74, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x69, 0x6e, 0x67, 0x2e, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0f, 0x74, 0x65, 0x78, 0x74, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x69,
	0x6e, 0x67, 0x2e, 0x76, 0x31, 0x1a, 0x1c, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x61, 0x70,
	0x69, 0x2f, 0x61, 0x6e, 0x6e, 0x6f, 0x74, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2e, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x22, 0x52, 0x0a, 0x0f, 0x45, 0x6e, 0x63, 0x6f, 0x64, 0x69, 0x6e, 0x67, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x14, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x29, 0x0a, 0x10,
	0x70, 0x6f, 0x6f, 0x6c, 0x69, 0x6e, 0x67, 0x5f, 0x73, 0x74, 0x72, 0x61, 0x74, 0x65, 0x67, 0x79,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x0f, 0x70, 0x6f, 0x6f, 0x6c, 0x69, 0x6e, 0x67, 0x53,
	0x74, 0x72, 0x61, 0x74, 0x65, 0x67, 0x79, 0x22, 0x2a, 0x0a, 0x10, 0x45, 0x6e, 0x63, 0x6f, 0x64,
	0x69, 0x6e, 0x67, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x16, 0x0a, 0x06, 0x76,
	0x65, 0x63, 0x74, 0x6f, 0x72, 0x18, 0x01, 0x20, 0x03, 0x28, 0x02, 0x52, 0x06, 0x76, 0x65, 0x63,
	0x74, 0x6f, 0x72, 0x32, 0x7b, 0x0a, 0x13, 0x54, 0x65, 0x78, 0x74, 0x45, 0x6e, 0x63, 0x6f, 0x64,
	0x69, 0x6e, 0x67, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x12, 0x64, 0x0a, 0x06, 0x45, 0x6e,
	0x63, 0x6f, 0x64, 0x65, 0x12, 0x20, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x65, 0x6e, 0x63, 0x6f, 0x64,
	0x69, 0x6e, 0x67, 0x2e, 0x76, 0x31, 0x2e, 0x45, 0x6e, 0x63, 0x6f, 0x64, 0x69, 0x6e, 0x67, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x21, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x65, 0x6e, 0x63,
	0x6f, 0x64, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x31, 0x2e, 0x45, 0x6e, 0x63, 0x6f, 0x64, 0x69, 0x6e,
	0x67, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x15, 0x82, 0xd3, 0xe4, 0x93, 0x02,
	0x0f, 0x22, 0x0a, 0x2f, 0x76, 0x31, 0x2f, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x65, 0x3a, 0x01, 0x2a,
	0x42, 0x50, 0x5a, 0x4e, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x6e,
	0x6c, 0x70, 0x6f, 0x64, 0x79, 0x73, 0x73, 0x65, 0x79, 0x2f, 0x63, 0x79, 0x62, 0x65, 0x72, 0x74,
	0x72, 0x6f, 0x6e, 0x2f, 0x70, 0x6b, 0x67, 0x2f, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2f, 0x61,
	0x70, 0x69, 0x73, 0x2f, 0x74, 0x65, 0x78, 0x74, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x69, 0x6e, 0x67,
	0x2f, 0x76, 0x31, 0x3b, 0x74, 0x65, 0x78, 0x74, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x69, 0x6e, 0x67,
	0x76, 0x31, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_textencoding_v1_textencoding_proto_rawDescOnce sync.Once
	file_textencoding_v1_textencoding_proto_rawDescData = file_textencoding_v1_textencoding_proto_rawDesc
)

func file_textencoding_v1_textencoding_proto_rawDescGZIP() []byte {
	file_textencoding_v1_textencoding_proto_rawDescOnce.Do(func() {
		file_textencoding_v1_textencoding_proto_rawDescData = protoimpl.X.CompressGZIP(file_textencoding_v1_textencoding_proto_rawDescData)
	})
	return file_textencoding_v1_textencoding_proto_rawDescData
}

var file_textencoding_v1_textencoding_proto_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_textencoding_v1_textencoding_proto_goTypes = []interface{}{
	(*EncodingRequest)(nil),  // 0: textencoding.v1.EncodingRequest
	(*EncodingResponse)(nil), // 1: textencoding.v1.EncodingResponse
}
var file_textencoding_v1_textencoding_proto_depIdxs = []int32{
	0, // 0: textencoding.v1.TextEncodingService.Encode:input_type -> textencoding.v1.EncodingRequest
	1, // 1: textencoding.v1.TextEncodingService.Encode:output_type -> textencoding.v1.EncodingResponse
	1, // [1:2] is the sub-list for method output_type
	0, // [0:1] is the sub-list for method input_type
	0, // [0:0] is the sub-list for extension type_name
	0, // [0:0] is the sub-list for extension extendee
	0, // [0:0] is the sub-list for field type_name
}

func init() { file_textencoding_v1_textencoding_proto_init() }
func file_textencoding_v1_textencoding_proto_init() {
	if File_textencoding_v1_textencoding_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_textencoding_v1_textencoding_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*EncodingRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_textencoding_v1_textencoding_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*EncodingResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_textencoding_v1_textencoding_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_textencoding_v1_textencoding_proto_goTypes,
		DependencyIndexes: file_textencoding_v1_textencoding_proto_depIdxs,
		MessageInfos:      file_textencoding_v1_textencoding_proto_msgTypes,
	}.Build()
	File_textencoding_v1_textencoding_proto = out.File
	file_textencoding_v1_textencoding_proto_rawDesc = nil
	file_textencoding_v1_textencoding_proto_goTypes = nil
	file_textencoding_v1_textencoding_proto_depIdxs = nil
}
