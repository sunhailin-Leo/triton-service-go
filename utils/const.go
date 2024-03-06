package utils

import (
	"errors"
	"unicode"
)

const (
	TritonBoolType   string = "BOOL"
	TritonUINT8Type  string = "UINT8"
	TritonUINT16Type string = "UINT16"
	TritonUINT32Type string = "UINT32"
	TritonUINT64Type string = "UINT64"
	TritonINT8Type   string = "INT8"
	TritonINT16Type  string = "INT16"
	TritonINT32Type  string = "INT32"
	TritonINT64Type  string = "INT64"
	TritonFP16Type   string = "FP16"
	TritonFP32Type   string = "FP32"
	TritonFP64Type   string = "FP64"
	TritonBytesType  string = "BYTES"
	TritonBF16Type   string = "BF16"
	SliceByteType    string = "[]byte"
	SliceFloat32Type string = "[]float32"
	SliceFloat64Type string = "[]float64"
	SliceIntType     string = "[]int"
	SliceInt64Type   string = "[]int64"
)

var (
	ErrEmptyVocab           = errors.New("empty vocab")              // empty vocab error.
	ErrEmptyCallbackFunc    = errors.New("callback function is nil") // empty callback function.
	ErrEmptyHTTPRequestBody = errors.New("http request body is nil") // empty http request body.
	ErrEmptyGRPCRequestBody = errors.New("grpc request body is nil") // empty grpc request body.

	// ASCIIWhiteSpace ascii white space array.
	ASCIIWhiteSpace = [256]bool{' ': true, '\t': true, '\n': true, '\r': true}

	// ASCIIPunctuation Ascii punctuation characters range.
	ASCIIPunctuation = &unicode.RangeTable{
		R16: []unicode.Range16{
			{0x0021, 0x002f, 1}, // 33-47
			{0x003a, 0x0040, 1}, // 58-64
			{0x005b, 0x0060, 1}, // 91-96
			{0x007b, 0x007e, 1}, // 123-126
		},
		LatinOffset: 4, // All less than 0x00FF
	}

	// BertChineseChar maybe is the BERT Chinese Char.
	BertChineseChar = &unicode.RangeTable{
		R16: []unicode.Range16{
			{0x4e00, 0x9fff, 1},
			{0x3400, 0x4dbf, 1},
			{0xf900, 0xfaff, 1},
		},
		R32: []unicode.Range32{
			{Lo: 0x20000, Hi: 0x2a6df, Stride: 1},
			{Lo: 0x2a700, Hi: 0x2b73f, Stride: 1},
			{Lo: 0x2b740, Hi: 0x2b81f, Stride: 1},
			{Lo: 0x2b820, Hi: 0x2ceaf, Stride: 1},
			{Lo: 0x2f800, Hi: 0x2fa1f, Stride: 1},
		},
	}
)
