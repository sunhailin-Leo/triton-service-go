package transformers_test

import (
	"encoding/binary"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
)

func testGenerateModelInferRequest() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor {
	return []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor{
		{Name: "segment_ids", Datatype: "INT32"},
		{Name: "input_ids", Datatype: "INT32"},
		{Name: "input_mask", Datatype: "INT32"},
	}
}

func testGenerateModelInferOutputRequest(params ...interface{}) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor {
	return []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor{
		{Name: "probability"},
	}
}

func testModelInferCallback(inferResponse interface{}, params ...interface{}) ([]interface{}, error) {
	return nil, nil
}

func newTestBertService(t *testing.T) *transformers.BertModelService {
	t.Helper()

	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt",
		"127.0.0.1:9001",
		nil,
		nil,
		testGenerateModelInferRequest,
		testGenerateModelInferOutputRequest,
		testModelInferCallback,
	)
	if err != nil {
		t.Fatalf("Failed to create test Bert service: %v", err)
	}
	return srv
}

func TestNewBertModelService_Success(t *testing.T) {
	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt",
		"127.0.0.1:9001",
		nil,
		nil,
		testGenerateModelInferRequest,
		testGenerateModelInferOutputRequest,
		testModelInferCallback,
	)

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if srv == nil {
		t.Fatal("Expected service to be created")
	}
	if srv.BertVocab.Size() == 0 {
		t.Error("Expected BertVocab to be initialized")
	}
	if srv.BertTokenizer == nil {
		t.Error("Expected BertTokenizer to be initialized")
	}
}

func TestNewBertModelService_NilCallbacks(t *testing.T) {
	tests := []struct {
		name           string
		inputCallback  func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor
		outputCallback func(...interface{}) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor
		inferCallback  nvidia_inferenceserver.DecoderFunc
		expectError    bool
	}{
		{
			name:           "nil input callback",
			inputCallback:  nil,
			outputCallback: testGenerateModelInferOutputRequest,
			inferCallback:  testModelInferCallback,
			expectError:    true,
		},
		{
			name:           "nil output callback",
			inputCallback:  testGenerateModelInferRequest,
			outputCallback: nil,
			inferCallback:  testModelInferCallback,
			expectError:    true,
		},
		{
			name:           "nil infer callback",
			inputCallback:  testGenerateModelInferRequest,
			outputCallback: testGenerateModelInferOutputRequest,
			inferCallback:  nil,
			expectError:    true,
		},
		{
			name:           "all nil callbacks",
			inputCallback:  nil,
			outputCallback: nil,
			inferCallback:  nil,
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := transformers.NewBertModelService(
				"../../test/bert-chinese-vocab.txt",
				"127.0.0.1:9001",
				nil,
				nil,
				tt.inputCallback,
				tt.outputCallback,
				tt.inferCallback,
			)

			if tt.expectError && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

func TestNewBertModelService_InvalidVocabPath(t *testing.T) {
	_, err := transformers.NewBertModelService(
		"../../test/non-existent-vocab.txt",
		"127.0.0.1:9001",
		nil,
		nil,
		testGenerateModelInferRequest,
		testGenerateModelInferOutputRequest,
		testModelInferCallback,
	)

	if err == nil {
		t.Error("Expected error for invalid vocab path, got nil")
	}
}

func TestBertModelService_SetChineseTokenize(t *testing.T) {
	srv := newTestBertService(t)

	tests := []struct {
		name             string
		isCharMode       bool
		expectedChinese  bool
		expectedCharMode bool
	}{
		{
			name:             "char mode true",
			isCharMode:       true,
			expectedChinese:  true,
			expectedCharMode: true,
		},
		{
			name:             "char mode false",
			isCharMode:       false,
			expectedChinese:  true,
			expectedCharMode: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv.SetChineseTokenize(tt.isCharMode)

			if srv.IsChinese != tt.expectedChinese {
				t.Errorf("Expected IsChinese=%v, got %v", tt.expectedChinese, srv.IsChinese)
			}
			if srv.IsChineseCharMode != tt.expectedCharMode {
				t.Errorf("Expected IsChineseCharMode=%v, got %v", tt.expectedCharMode, srv.IsChineseCharMode)
			}
		})
	}
}

func TestBertModelService_UnsetChineseTokenize(t *testing.T) {
	srv := newTestBertService(t)

	srv.SetChineseTokenize(true)
	srv.UnsetChineseTokenize()

	if srv.IsChinese != false {
		t.Errorf("Expected IsChinese=false, got %v", srv.IsChinese)
	}
	if srv.IsChineseCharMode != false {
		t.Errorf("Expected IsChineseCharMode=false, got %v", srv.IsChineseCharMode)
	}
}

func TestBertModelService_SetModelInferWithGRPC(t *testing.T) {
	srv := newTestBertService(t)

	srv.SetModelInferWithGRPC()

	if srv.IsGRPC != true {
		t.Errorf("Expected IsGRPC=true, got %v", srv.IsGRPC)
	}
}

func TestBertModelService_UnsetModelInferWithGRPC(t *testing.T) {
	srv := newTestBertService(t)

	srv.SetModelInferWithGRPC()
	srv.UnsetModelInferWithGRPC()

	if srv.IsGRPC != false {
		t.Errorf("Expected IsGRPC=false, got %v", srv.IsGRPC)
	}
}

func TestBertModelService_SetMaxSeqLength(t *testing.T) {
	srv := newTestBertService(t)

	tests := []struct {
		name     string
		maxLen   int
		expected int
	}{
		{
			name:     "set to 64",
			maxLen:   64,
			expected: 64,
		},
		{
			name:     "set to 128",
			maxLen:   128,
			expected: 128,
		},
		{
			name:     "set to 512",
			maxLen:   512,
			expected: 512,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv.SetMaxSeqLength(tt.maxLen)

			if srv.MaxSeqLength != tt.expected {
				t.Errorf("Expected MaxSeqLength=%d, got %d", tt.expected, srv.MaxSeqLength)
			}
		})
	}
}

func TestBertModelService_SetModelName(t *testing.T) {
	srv := newTestBertService(t)

	srv.SetModelName("bert", "base-chinese")

	expected := "bert-base-chinese"
	if srv.ModelName != expected {
		t.Errorf("Expected ModelName=%s, got %s", expected, srv.ModelName)
	}
}

func TestBertModelService_SetModelNameWithoutDash(t *testing.T) {
	srv := newTestBertService(t)

	srv.SetModelNameWithoutDash("bert_base_chinese")

	expected := "bert_base_chinese"
	if srv.ModelName != expected {
		t.Errorf("Expected ModelName=%s, got %s", expected, srv.ModelName)
	}
}

func TestBertModelService_GetModelName(t *testing.T) {
	srv := newTestBertService(t)

	srv.SetModelName("bert", "base-chinese")

	name := srv.GetModelName()

	expected := "bert-base-chinese"
	if name != expected {
		t.Errorf("Expected GetModelName()=%s, got %s", expected, name)
	}
}

func TestBertModelService_SetTokenizerReturnPosInfo(t *testing.T) {
	srv := newTestBertService(t)

	srv.SetTokenizerReturnPosInfo()

	if srv.IsReturnPosArray != true {
		t.Errorf("Expected IsReturnPosArray=true, got %v", srv.IsReturnPosArray)
	}
}

func TestBertModelService_UnsetTokenizerReturnPosInfo(t *testing.T) {
	srv := newTestBertService(t)

	srv.SetTokenizerReturnPosInfo()
	srv.UnsetTokenizerReturnPosInfo()

	if srv.IsReturnPosArray != false {
		t.Errorf("Expected IsReturnPosArray=false, got %v", srv.IsReturnPosArray)
	}
}

func TestBertModelService_GrpcSliceToLittleEndianByteSlice_Int32(t *testing.T) {
	srv := newTestBertService(t)

	input := []int32{1, 2, 3, 4, 5}
	result := srv.GrpcSliceToLittleEndianByteSlice(5, input, "INT32")

	if result == nil {
		t.Fatal("Expected non-nil result for INT32")
	}
	if len(result) != 20 {
		t.Errorf("Expected result length 20 (5*4), got %d", len(result))
	}

	for i, val := range input {
		decoded := int32(binary.LittleEndian.Uint32(result[i*4 : (i+1)*4]))
		if decoded != val {
			t.Errorf("At index %d: expected %d, got %d", i, val, decoded)
		}
	}
}

func TestBertModelService_GrpcSliceToLittleEndianByteSlice_Int64(t *testing.T) {
	srv := newTestBertService(t)

	input := []int32{1, 2, 3, 4, 5}
	result := srv.GrpcSliceToLittleEndianByteSlice(5, input, "INT64")

	if result == nil {
		t.Fatal("Expected non-nil result for INT64")
	}
	if len(result) != 40 {
		t.Errorf("Expected result length 40 (5*8), got %d", len(result))
	}

	for i, val := range input {
		decoded := int64(binary.LittleEndian.Uint64(result[i*8 : (i+1)*8]))
		if decoded != int64(val) {
			t.Errorf("At index %d: expected %d, got %d", i, val, decoded)
		}
	}
}

func TestBertModelService_GrpcSliceToLittleEndianByteSlice_Bool(t *testing.T) {
	srv := newTestBertService(t)

	input := []bool{true, false, true, false, true}
	result := srv.GrpcSliceToLittleEndianByteSlice(5, input, "BOOL")

	if result == nil {
		t.Fatal("Expected non-nil result for BOOL")
	}
	if len(result) != 5 {
		t.Errorf("Expected result length 5, got %d", len(result))
	}

	expected := []byte{1, 0, 1, 0, 1}
	for i, val := range result {
		if val != expected[i] {
			t.Errorf("At index %d: expected %d, got %d", i, expected[i], val)
		}
	}
}

func TestBertModelService_GrpcSliceToLittleEndianByteSlice_UnsupportedType(t *testing.T) {
	srv := newTestBertService(t)

	input := []string{"a", "b", "c"}
	result := srv.GrpcSliceToLittleEndianByteSlice(3, input, "INT32")

	if result != nil {
		t.Error("Expected nil result for unsupported type")
	}
}

func TestBertModelService_GrpcSliceToLittleEndianByteSlice_UnsupportedDatatype(t *testing.T) {
	srv := newTestBertService(t)

	input := []int32{1, 2, 3}
	result := srv.GrpcSliceToLittleEndianByteSlice(3, input, "FLOAT32")

	if result != nil {
		t.Error("Expected nil result for unsupported datatype")
	}
}

func BenchmarkGrpcSliceToLittleEndianByteSlice_Int32(b *testing.B) {
	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt",
		"127.0.0.1:9001",
		nil,
		nil,
		testGenerateModelInferRequest,
		testGenerateModelInferOutputRequest,
		testModelInferCallback,
	)
	if err != nil {
		b.Fatal(err)
	}

	input := make([]int32, 128)
	for i := range input {
		input[i] = int32(i)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = srv.GrpcSliceToLittleEndianByteSlice(128, input, "INT32")
	}
}

func BenchmarkGrpcSliceToLittleEndianByteSlice_Bool(b *testing.B) {
	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt",
		"127.0.0.1:9001",
		nil,
		nil,
		testGenerateModelInferRequest,
		testGenerateModelInferOutputRequest,
		testModelInferCallback,
	)
	if err != nil {
		b.Fatal(err)
	}

	input := make([]bool, 128)
	for i := range input {
		input[i] = i%2 == 0
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = srv.GrpcSliceToLittleEndianByteSlice(128, input, "BOOL")
	}
}
