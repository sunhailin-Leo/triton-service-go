package test

import (
	"encoding/binary"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func newTestBertService(t *testing.T) *transformers.BertModelService {
	t.Helper()
	vocabPath := "bert-chinese-vocab.txt"
	httpAddr := "127.0.0.1:9001"
	grpcAddr := "127.0.0.1:9000"
	defaultHTTPClient := &fasthttp.Client{}
	defaultGRPCClient, grpcErr := grpc.NewClient(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if grpcErr != nil {
		t.Fatalf("failed to create grpc client: %v", grpcErr)
	}

	bertService, initErr := transformers.NewBertModelService(
		vocabPath, httpAddr, defaultHTTPClient, defaultGRPCClient,
		testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback)
	if initErr != nil {
		t.Fatalf("failed to init bert service: %v", initErr)
	}
	bertService.SetChineseTokenize(false).SetMaxSeqLength(48)
	return bertService
}

// TestNewBertModelService_NilCallbacks tests creating bert service with nil callbacks
func TestNewBertModelService_NilCallbacks(t *testing.T) {
	vocabPath := "bert-chinese-vocab.txt"
	httpAddr := "127.0.0.1:9001"

	_, err := transformers.NewBertModelService(
		vocabPath, httpAddr, nil, nil,
		nil, nil, nil)
	if err == nil {
		t.Fatal("expected error when passing nil callbacks")
	}
}

// TestNewBertModelService_InvalidVocabPath tests creating bert service with invalid vocab path
func TestNewBertModelService_InvalidVocabPath(t *testing.T) {
	_, err := transformers.NewBertModelService(
		"nonexistent-vocab.txt", "127.0.0.1:9001", nil, nil,
		testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback)
	if err == nil {
		t.Fatal("expected error when passing invalid vocab path")
	}
}

// TestBertModelService_SetModelName tests model name setting
func TestBertModelService_SetModelName(t *testing.T) {
	bertService := newTestBertService(t)

	bertService.SetModelName("prefix", "name")
	if bertService.GetModelName() != "prefix-name" {
		t.Errorf("expected model name 'prefix-name', got '%s'", bertService.GetModelName())
	}

	bertService.SetModelNameWithoutDash("mymodel")
	if bertService.GetModelName() != "mymodel" {
		t.Errorf("expected model name 'mymodel', got '%s'", bertService.GetModelName())
	}
}

// TestBertModelService_Flags tests flag setters/getters
func TestBertModelService_Flags(t *testing.T) {
	bertService := newTestBertService(t)

	// Test GRPC flag
	bertService.SetModelInferWithGRPC()
	if !bertService.GetModelInferIsGRPC() {
		t.Error("expected IsGRPC to be true")
	}
	bertService.UnsetModelInferWithGRPC()
	if bertService.GetModelInferIsGRPC() {
		t.Error("expected IsGRPC to be false")
	}

	// Test Chinese tokenizer flag
	bertService.SetChineseTokenize(true)
	if !bertService.GetTokenizerIsChineseMode() {
		t.Error("expected IsChinese to be true")
	}
	bertService.UnsetChineseTokenize()
	if bertService.GetTokenizerIsChineseMode() {
		t.Error("expected IsChinese to be false")
	}
}

// TestGrpcSliceToLittleEndianByteSlice_Int32 tests int32 conversion
func TestGrpcSliceToLittleEndianByteSlice_Int32(t *testing.T) {
	bertService := newTestBertService(t)

	input := []int32{1, 2, 3}
	result := bertService.GrpcSliceToLittleEndianByteSlice(3, input, "INT32")

	if len(result) != 12 { // 3 * 4 bytes
		t.Fatalf("expected 12 bytes, got %d", len(result))
	}

	for i, expected := range input {
		got := int32(binary.LittleEndian.Uint32(result[i*4:]))
		if got != expected {
			t.Errorf("at index %d: expected %d, got %d", i, expected, got)
		}
	}
}

// TestGrpcSliceToLittleEndianByteSlice_Int64 tests int64 conversion
func TestGrpcSliceToLittleEndianByteSlice_Int64(t *testing.T) {
	bertService := newTestBertService(t)

	input := []int32{100, 200, 300}
	result := bertService.GrpcSliceToLittleEndianByteSlice(3, input, "INT64")

	if len(result) != 24 { // 3 * 8 bytes
		t.Fatalf("expected 24 bytes, got %d", len(result))
	}

	for i, expected := range input {
		got := int64(binary.LittleEndian.Uint64(result[i*8:]))
		if got != int64(expected) {
			t.Errorf("at index %d: expected %d, got %d", i, expected, got)
		}
	}
}

// TestGrpcSliceToLittleEndianByteSlice_Bool tests bool conversion
func TestGrpcSliceToLittleEndianByteSlice_Bool(t *testing.T) {
	bertService := newTestBertService(t)

	input := []bool{true, false, true, false}
	result := bertService.GrpcSliceToLittleEndianByteSlice(4, input, "BOOL")

	if len(result) != 4 {
		t.Fatalf("expected 4 bytes, got %d", len(result))
	}

	expected := []byte{1, 0, 1, 0}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("at index %d: expected %d, got %d", i, expected[i], result[i])
		}
	}
}

// TestGrpcSliceToLittleEndianByteSlice_UnsupportedType tests unsupported type
func TestGrpcSliceToLittleEndianByteSlice_UnsupportedType(t *testing.T) {
	bertService := newTestBertService(t)

	// Unsupported input type
	result := bertService.GrpcSliceToLittleEndianByteSlice(3, "invalid", "INT32")
	if result != nil {
		t.Error("expected nil for unsupported type")
	}

	// Unsupported data type for int32 slice
	input := []int32{1, 2, 3}
	result = bertService.GrpcSliceToLittleEndianByteSlice(3, input, "UNKNOWN")
	if result != nil {
		t.Error("expected nil for unknown data type")
	}
}

// BenchmarkGrpcSliceToLittleEndianByteSlice_Int32 benchmarks int32 conversion
func BenchmarkGrpcSliceToLittleEndianByteSlice_Int32(b *testing.B) {
	vocabPath := "bert-chinese-vocab.txt"
	httpAddr := "127.0.0.1:9001"
	grpcAddr := "127.0.0.1:9000"
	defaultHTTPClient := &fasthttp.Client{}
	defaultGRPCClient, _ := grpc.NewClient(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))

	bertService, _ := transformers.NewBertModelService(
		vocabPath, httpAddr, defaultHTTPClient, defaultGRPCClient,
		testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback)

	input := make([]int32, 128)
	for i := range input {
		input[i] = int32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bertService.GrpcSliceToLittleEndianByteSlice(128, input, "INT32")
	}
	b.ReportAllocs()
}

// BenchmarkGrpcSliceToLittleEndianByteSlice_Bool benchmarks bool conversion
func BenchmarkGrpcSliceToLittleEndianByteSlice_Bool(b *testing.B) {
	vocabPath := "bert-chinese-vocab.txt"
	httpAddr := "127.0.0.1:9001"
	grpcAddr := "127.0.0.1:9000"
	defaultHTTPClient := &fasthttp.Client{}
	defaultGRPCClient, _ := grpc.NewClient(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))

	bertService, _ := transformers.NewBertModelService(
		vocabPath, httpAddr, defaultHTTPClient, defaultGRPCClient,
		testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback)

	input := make([]bool, 128)
	for i := range input {
		input[i] = i%2 == 0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bertService.GrpcSliceToLittleEndianByteSlice(128, input, "BOOL")
	}
	b.ReportAllocs()
}
