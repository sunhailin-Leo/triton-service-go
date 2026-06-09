package transformers_test

import (
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
)

// --- BertOption tests ---

func TestBertOption_WithMaxSeqLength(t *testing.T) {
	opt := transformers.WithBertMaxSeqLength(128)
	srv := &transformers.BertModelService{}
	opt(srv)
	if srv.MaxSeqLength != 128 {
		t.Errorf("expected MaxSeqLength 128, got %d", srv.MaxSeqLength)
	}
}

func TestBertOption_WithChineseTokenize(t *testing.T) {
	opt := transformers.WithBertChineseTokenize(true)
	srv := &transformers.BertModelService{}
	opt(srv)
	if !srv.IsChinese {
		t.Error("expected IsChinese to be true")
	}
	if !srv.IsChineseCharMode {
		t.Error("expected IsChineseCharMode to be true")
	}
}

func TestBertOption_WithChineseTokenize_FalseCharMode(t *testing.T) {
	opt := transformers.WithBertChineseTokenize(false)
	srv := &transformers.BertModelService{}
	opt(srv)
	if !srv.IsChinese {
		t.Error("expected IsChinese to be true")
	}
	if srv.IsChineseCharMode {
		t.Error("expected IsChineseCharMode to be false")
	}
}

func TestBertOption_WithGRPCInfer(t *testing.T) {
	opt := transformers.WithBertGRPCInfer()
	srv := &transformers.BertModelService{}
	opt(srv)
	if !srv.IsGRPC {
		t.Error("expected IsGRPC to be true")
	}
}

func TestBertOption_WithTokenizerReturnPosInfo(t *testing.T) {
	opt := transformers.WithBertTokenizerReturnPosInfo()
	srv := &transformers.BertModelService{}
	opt(srv)
	if !srv.IsReturnPosArray {
		t.Error("expected IsReturnPosArray to be true")
	}
}

func TestBertOption_WithModelName(t *testing.T) {
	opt := transformers.WithBertModelName("bert", "base")
	srv := &transformers.BertModelService{}
	opt(srv)
	if srv.ModelName != "bert-base" {
		t.Errorf("expected ModelName 'bert-base', got %q", srv.ModelName)
	}
}

func TestBertOption_WithModelNameWithoutDash(t *testing.T) {
	opt := transformers.WithBertModelNameWithoutDash("mymodel")
	srv := &transformers.BertModelService{}
	opt(srv)
	if srv.ModelName != "mymodel" {
		t.Errorf("expected ModelName 'mymodel', got %q", srv.ModelName)
	}
}

func TestBertOption_Multiple(t *testing.T) {
	srv := &transformers.BertModelService{}
	transformers.WithBertMaxSeqLength(256)(srv)
	transformers.WithBertChineseTokenize(false)(srv)
	transformers.WithBertGRPCInfer()(srv)
	transformers.WithBertTokenizerReturnPosInfo()(srv)
	transformers.WithBertModelName("bert", "large")(srv)

	if srv.MaxSeqLength != 256 {
		t.Errorf("expected MaxSeqLength 256, got %d", srv.MaxSeqLength)
	}
	if !srv.IsChinese {
		t.Error("expected IsChinese to be true")
	}
	if !srv.IsGRPC {
		t.Error("expected IsGRPC to be true")
	}
	if !srv.IsReturnPosArray {
		t.Error("expected IsReturnPosArray to be true")
	}
	if srv.ModelName != "bert-large" {
		t.Errorf("expected ModelName 'bert-large', got %q", srv.ModelName)
	}
}

func newTestW2NerService() *transformers.W2NerModelService {
	cli := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	bertSrv, _ := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil },
		func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor { return nil },
		func(response any, params ...any) ([]any, error) { return nil, nil },
	)
	_ = cli
	return &transformers.W2NerModelService{BertModelService: bertSrv}
}

// --- W2NerOption tests ---

func TestW2NerOption_WithMaxSeqLength(t *testing.T) {
	srv := newTestW2NerService()
	opt := transformers.WithW2NerMaxSeqLength(128)
	opt(srv)
	if srv.MaxSeqLength != 128 {
		t.Errorf("expected MaxSeqLength 128, got %d", srv.MaxSeqLength)
	}
}

func TestW2NerOption_WithChineseTokenize(t *testing.T) {
	srv := newTestW2NerService()
	opt := transformers.WithW2NerChineseTokenize(true)
	opt(srv)
	if !srv.IsChinese {
		t.Error("expected IsChinese to be true")
	}
	if !srv.IsChineseCharMode {
		t.Error("expected IsChineseCharMode to be true")
	}
}

func TestW2NerOption_WithGRPCInfer(t *testing.T) {
	srv := newTestW2NerService()
	opt := transformers.WithW2NerGRPCInfer()
	opt(srv)
	if !srv.IsGRPC {
		t.Error("expected IsGRPC to be true")
	}
}

func TestW2NerOption_WithTokenizerReturnPosInfo(t *testing.T) {
	srv := newTestW2NerService()
	opt := transformers.WithW2NerTokenizerReturnPosInfo()
	opt(srv)
	if !srv.IsReturnPosArray {
		t.Error("expected IsReturnPosArray to be true")
	}
}

func TestW2NerOption_WithModelName(t *testing.T) {
	srv := newTestW2NerService()
	opt := transformers.WithW2NerModelName("w2ner", "base")
	opt(srv)
	if srv.ModelName != "w2ner-base" {
		t.Errorf("expected ModelName 'w2ner-base', got %q", srv.ModelName)
	}
}

// --- GrpcSliceToLittleEndianByteSlice tests ---

func TestGrpcSliceToLittleEndianByteSlice_Int32(t *testing.T) {
	srv := &transformers.BertModelService{}
	input := []int32{1, 2, 3}
	result := srv.GrpcSliceToLittleEndianByteSlice(3, input, "INT32")
	if len(result) != 12 { // 3 * 4 bytes
		t.Errorf("expected 12 bytes, got %d", len(result))
	}
}

func TestGrpcSliceToLittleEndianByteSlice_Int64(t *testing.T) {
	srv := &transformers.BertModelService{}
	input := []int32{1, 2}
	result := srv.GrpcSliceToLittleEndianByteSlice(2, input, "INT64")
	if len(result) != 16 { // 2 * 8 bytes
		t.Errorf("expected 16 bytes, got %d", len(result))
	}
}

func TestGrpcSliceToLittleEndianByteSlice_Bool(t *testing.T) {
	srv := &transformers.BertModelService{}
	input := []bool{true, false, true}
	result := srv.GrpcSliceToLittleEndianByteSlice(3, input, "BOOL")
	if len(result) != 3 {
		t.Errorf("expected 3 bytes, got %d", len(result))
	}
	if result[0] != 1 || result[1] != 0 || result[2] != 1 {
		t.Errorf("expected [1, 0, 1], got %v", result)
	}
}

func TestGrpcSliceToLittleEndianByteSlice_UnsupportedType(t *testing.T) {
	srv := &transformers.BertModelService{}
	input := []int32{1, 2}
	result := srv.GrpcSliceToLittleEndianByteSlice(2, input, "FLOAT32")
	if result != nil {
		t.Errorf("expected nil for unsupported type, got %v", result)
	}
}

func TestGrpcSliceToLittleEndianByteSlice_UnsupportedInputType(t *testing.T) {
	srv := &transformers.BertModelService{}
	input := []float64{1.0, 2.0}
	result := srv.GrpcSliceToLittleEndianByteSlice(2, input, "INT32")
	if result != nil {
		t.Errorf("expected nil for unsupported input type, got %v", result)
	}
}

// --- VocabFromSlice detailed test ---

func TestVocabFromSlice_LargeSlice(t *testing.T) {
	tokens := make([]string, 10000)
	for i := range tokens {
		tokens[i] = "token_" + string(rune(i))
	}
	vocab, err := transformers.VocabFromSlice(tokens)
	if err != nil {
		t.Fatalf("VocabFromSlice failed: %v", err)
	}
	if vocab.Size() != 10000 {
		t.Errorf("expected size 10000, got %d", vocab.Size())
	}
}

// --- IsDefaultSpecial table-driven tests ---

func TestIsDefaultSpecial_Table(t *testing.T) {
	tests := []struct {
		word     string
		expected bool
	}{
		{"[UNK]", true},
		{"[CLS]", true},
		{"[SEP]", true},
		{"[MASK]", true},
		{"hello", false},
		{"", false},
		{"unk", false},
		{"[SPECIAL]", false},
	}
	for _, tt := range tests {
		t.Run(tt.word, func(t *testing.T) {
			if got := transformers.IsDefaultSpecial(tt.word); got != tt.expected {
				t.Errorf("IsDefaultSpecial(%q) = %v, want %v", tt.word, got, tt.expected)
			}
		})
	}
}

// --- GroupPieces extended tests ---

func TestGroupPieces_Extended(t *testing.T) {
	tests := []struct {
		name    string
		tokens  []transformers.StringOffsetsPair
		wantLen int
	}{
		{
			name: "single tokens only",
			tokens: []transformers.StringOffsetsPair{
				{String: "hello", Offsets: transformers.OffsetsType{Start: 0, End: 5}},
				{String: "world", Offsets: transformers.OffsetsType{Start: 6, End: 11}},
			},
			wantLen: 2,
		},
		{
			name: "subword tokens",
			tokens: []transformers.StringOffsetsPair{
				{String: "un", Offsets: transformers.OffsetsType{Start: 0, End: 2}},
				{String: "##afford", Offsets: transformers.OffsetsType{Start: 2, End: 8}},
				{String: "##able", Offsets: transformers.OffsetsType{Start: 8, End: 12}},
			},
			wantLen: 1,
		},
		{
			name: "mixed tokens",
			tokens: []transformers.StringOffsetsPair{
				{String: "hello", Offsets: transformers.OffsetsType{Start: 0, End: 5}},
				{String: "##wor", Offsets: transformers.OffsetsType{Start: 5, End: 8}},
				{String: "##ld", Offsets: transformers.OffsetsType{Start: 8, End: 10}},
				{String: "test", Offsets: transformers.OffsetsType{Start: 11, End: 15}},
			},
			wantLen: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			groups := transformers.GroupPieces(tt.tokens)
			if len(groups) != tt.wantLen {
				t.Errorf("GroupPieces() returned %d groups, want %d", len(groups), tt.wantLen)
			}
		})
	}
}

// --- NewBertModelService validation test ---

func TestNewBertModelService_NilCallback(t *testing.T) {
	cli := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	// Nil input callback
	_, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt", "127.0.0.1:9001", nil, nil,
		nil, nil, nil,
		transformers.WithBertMaxSeqLength(48),
	)
	_ = cli
	if err == nil {
		t.Error("expected error for nil callback functions")
	}
}
