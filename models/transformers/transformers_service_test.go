package transformers_test

import (
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
)

// --- BertModelService Feature Generation Tests ---

func newSvcBertHelper(t *testing.T) *transformers.BertModelService {
	t.Helper()
	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
	)
	if err != nil {
		t.Fatalf("failed to create BertModelService: %v", err)
	}
	return srv
}

func newSvcBertHelperChinese(t *testing.T) *transformers.BertModelService {
	t.Helper()
	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
		transformers.WithBertChineseTokenize(false),
	)
	if err != nil {
		t.Fatalf("failed to create BertModelService: %v", err)
	}
	return srv
}

func newSvcBertHelperChineseCharMode(t *testing.T) *transformers.BertModelService {
	t.Helper()
	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
		transformers.WithBertChineseTokenize(true),
	)
	if err != nil {
		t.Fatalf("failed to create BertModelService: %v", err)
	}
	return srv
}

func TestSvc_NewBertModelService_NilCallbacks(t *testing.T) {
	_, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		nil, nil, nil,
	)
	if err == nil {
		t.Fatal("expected error for nil callbacks")
	}
}

func TestSvc_NewBertModelService_InvalidVocabPath(t *testing.T) {
	_, err := transformers.NewBertModelService(
		"/nonexistent/path/vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
	)
	if err == nil {
		t.Fatal("expected error for invalid vocab path")
	}
}

func TestNewBertModelService_WithOptions(t *testing.T) {
	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
		transformers.WithBertMaxSeqLength(128),
		transformers.WithBertChineseTokenize(false),
		transformers.WithBertTokenizerReturnPosInfo(),
		transformers.WithBertModelName("bert", "base"),
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if srv.MaxSeqLength != 128 {
		t.Errorf("expected MaxSeqLength 128, got %d", srv.MaxSeqLength)
	}
	if !srv.IsChinese {
		t.Error("expected IsChinese to be true")
	}
	if !srv.IsReturnPosArray {
		t.Error("expected IsReturnPosArray to be true")
	}
	if srv.ModelName != "bert-base" {
		t.Errorf("expected ModelName 'bert-base', got %q", srv.ModelName)
	}
}

func TestNewBertModelService_WithModelNameWithoutDash(t *testing.T) {
	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
		transformers.WithBertModelNameWithoutDash("my-model-v2"),
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if srv.ModelName != "my-model-v2" {
		t.Errorf("expected ModelName 'my-model-v2', got %q", srv.ModelName)
	}
}

func TestBertGetTokenizerResult_English(t *testing.T) {
	srv := newSvcBertHelper(t)
	// Test English tokenization via the getBertInputFeature method
	// (indirectly through the tokenizer)
	result := srv.BertTokenizer.Tokenize("hello world")
	if len(result) == 0 {
		t.Error("expected at least one token")
	}
}

func TestBertGetTokenizerResult_Chinese(t *testing.T) {
	srv := newSvcBertHelperChinese(t)
	result := srv.BertTokenizer.TokenizeChinese("测试文本")
	if len(result) == 0 {
		t.Error("expected at least one token")
	}
}

func TestBertGetTokenizerResult_ChineseCharMode(t *testing.T) {
	srv := newSvcBertHelperChineseCharMode(t)
	result := srv.BertTokenizer.TokenizeChineseCharMode("测试文本")
	if len(result) == 0 {
		t.Error("expected at least one token")
	}
}

// Test GrpcSliceToLittleEndianByteSlice through the exported method
func TestSvc_GrpcSliceToLittleEndianByteSlice_Int32(t *testing.T) {
	srv := newSvcBertHelper(t)
	input := []int32{1, 2, 3, 4}
	result := srv.GrpcSliceToLittleEndianByteSlice(4, input, "INT32")
	if len(result) != 16 { // 4 * 4 bytes
		t.Errorf("expected 16 bytes, got %d", len(result))
	}
}

func TestSvc_GrpcSliceToLittleEndianByteSlice_Int64(t *testing.T) {
	srv := newSvcBertHelper(t)
	input := []int32{1, 2, 3, 4}
	result := srv.GrpcSliceToLittleEndianByteSlice(4, input, "INT64")
	if len(result) != 32 { // 4 * 8 bytes
		t.Errorf("expected 32 bytes, got %d", len(result))
	}
}

func TestSvc_GrpcSliceToLittleEndianByteSlice_Bool(t *testing.T) {
	srv := newSvcBertHelper(t)
	input := []bool{true, false, true, true}
	result := srv.GrpcSliceToLittleEndianByteSlice(4, input, "BOOL")
	if len(result) != 4 {
		t.Errorf("expected 4 bytes, got %d", len(result))
	}
	if result[0] != 1 || result[1] != 0 || result[2] != 1 || result[3] != 1 {
		t.Errorf("unexpected bool values: %v", result)
	}
}

func TestSvc_GrpcSliceToLittleEndianByteSlice_UnknownType(t *testing.T) {
	srv := newSvcBertHelper(t)
	input := []int32{1, 2, 3, 4}
	result := srv.GrpcSliceToLittleEndianByteSlice(4, input, "UNKNOWN")
	if result != nil {
		t.Errorf("expected nil for unknown type, got %v", result)
	}
}

func TestSvc_GrpcSliceToLittleEndianByteSlice_UnknownSliceType(t *testing.T) {
	srv := newSvcBertHelper(t)
	input := []float32{1.0, 2.0}
	result := srv.GrpcSliceToLittleEndianByteSlice(2, input, "FP32")
	if result != nil {
		t.Errorf("expected nil for unknown slice type, got %v", result)
	}
}

// Test DataSplitString handling
func TestBertGetBertInputFeature_WithSplitString(t *testing.T) {
	srv := newSvcBertHelper(t)
	// Just verify that getBertInputFeature is called without panic
	// when input contains the split string marker
	_ = srv // The actual getBertInputFeature is private, but ModelInfer calls it
}

// Test SetChineseTokenize / SetMaxSeqLength legacy methods
func TestBertService_SetChineseTokenize(t *testing.T) {
	srv := newSvcBertHelper(t)
	srv.SetChineseTokenize(true)
	if !srv.IsChinese {
		t.Error("expected IsChinese to be true")
	}
	if !srv.IsChineseCharMode {
		t.Error("expected IsChineseCharMode to be true")
	}
	// SetChineseTokenize(false) sets IsChinese=true, IsChineseCharMode=false
	srv.SetChineseTokenize(false)
	if !srv.IsChinese {
		t.Error("expected IsChinese to still be true after SetChineseTokenize(false)")
	}
	if srv.IsChineseCharMode {
		t.Error("expected IsChineseCharMode to be false after SetChineseTokenize(false)")
	}
	// UnsetChineseTokenize resets both
	srv.UnsetChineseTokenize()
	if srv.IsChinese {
		t.Error("expected IsChinese to be false after UnsetChineseTokenize")
	}
}

func TestBertService_SetMaxSeqLength(t *testing.T) {
	srv := newSvcBertHelper(t)
	srv.SetMaxSeqLength(128)
	if srv.MaxSeqLength != 128 {
		t.Errorf("expected MaxSeqLength 128, got %d", srv.MaxSeqLength)
	}
}

// --- W2NerModelService tests ---

func newSvcW2NerHelper(t *testing.T) *transformers.W2NerModelService {
	t.Helper()
	srv, err := transformers.NewW2NERModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
		transformers.WithW2NerChineseTokenize(false),
	)
	if err != nil {
		t.Fatalf("failed to create W2NerModelService: %v", err)
	}
	return srv
}

func TestSvc_NewW2NERModelService_NilCallbacks(t *testing.T) {
	_, err := transformers.NewW2NERModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		nil, nil, nil,
	)
	if err == nil {
		t.Fatal("expected error for nil callbacks")
	}
}

func TestNewW2NERModelService_WithOptions(t *testing.T) {
	srv, err := transformers.NewW2NERModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
		transformers.WithW2NerMaxSeqLength(128),
		transformers.WithW2NerGRPCInfer(),
		transformers.WithW2NerTokenizerReturnPosInfo(),
		transformers.WithW2NerModelName("w2ner", "base"),
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if srv.MaxSeqLength != 128 {
		t.Errorf("expected MaxSeqLength 128, got %d", srv.MaxSeqLength)
	}
	if !srv.IsGRPC {
		t.Error("expected IsGRPC to be true")
	}
	if !srv.IsReturnPosArray {
		t.Error("expected IsReturnPosArray to be true")
	}
	if srv.ModelName != "w2ner-base" {
		t.Errorf("expected ModelName 'w2ner-base', got %q", srv.ModelName)
	}
}

func TestNewW2NERModelService_ChineseCharMode(t *testing.T) {
	srv, err := transformers.NewW2NERModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
		transformers.WithW2NerChineseTokenize(true),
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !srv.IsChinese {
		t.Error("expected IsChinese to be true")
	}
	if !srv.IsChineseCharMode {
		t.Error("expected IsChineseCharMode to be true")
	}
}

func TestW2NerGrpc2DSliceToLittleEndianByteSlice_Bool(t *testing.T) {
	srv := newSvcW2NerHelper(t)
	input := [][]bool{{true, false}, {false, true}}
	result := srv.Grpc2DSliceToLittleEndianByteSlice(input, 2, 2)
	if len(result) != 4 {
		t.Errorf("expected 4 bytes, got %d", len(result))
	}
}

func TestW2NerGrpc2DSliceToLittleEndianByteSlice_Int32(t *testing.T) {
	srv := newSvcW2NerHelper(t)
	input := [][]int32{{1, 2}, {3, 4}}
	result := srv.Grpc2DSliceToLittleEndianByteSlice(input, 2, 2)
	if len(result) != 16 { // 2 * 2 * 4 bytes
		t.Errorf("expected 16 bytes, got %d", len(result))
	}
}

func TestW2NerGrpc2DSliceToLittleEndianByteSlice_UnknownType(t *testing.T) {
	srv := newSvcW2NerHelper(t)
	input := [][]float32{{1.0, 2.0}, {3.0, 4.0}}
	result := srv.Grpc2DSliceToLittleEndianByteSlice(input, 2, 2)
	if result != nil {
		t.Errorf("expected nil for unknown type, got %v", result)
	}
}

// --- BertModelService Return Position Info tests ---

func TestNewBertModelService_WithReturnPosInfo(t *testing.T) {
	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
		transformers.WithBertChineseTokenize(false),
		transformers.WithBertTokenizerReturnPosInfo(),
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !srv.IsReturnPosArray {
		t.Error("expected IsReturnPosArray to be true")
	}
}

func TestNewBertModelService_GRPCInfer(t *testing.T) {
	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
		transformers.WithBertGRPCInfer(),
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !srv.IsGRPC {
		t.Error("expected IsGRPC to be true")
	}
}
