package transformers_test

import (
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
)

// Helper function to create a BertModelService for testing.
func newTestBertModelService(t *testing.T) *transformers.BertModelService {
	t.Helper()
	modelInputCallback := func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor { return nil }
	modelOutputCallback := func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor {
		return nil
	}
	modelInferCallback := func(response any, params ...any) ([]any, error) { return nil, nil }

	srv, err := transformers.NewBertModelService(
		"../../test/bert-chinese-vocab.txt",
		"http://localhost:8000",
		nil,
		nil,
		modelInputCallback,
		modelOutputCallback,
		modelInferCallback,
	)
	if err != nil {
		t.Fatalf("Failed to create BertModelService: %v", err)
	}
	return srv
}

func TestBertModelService_GetBertInputFeature(t *testing.T) {
	srv := newTestBertModelService(t)

	tests := []struct {
		name      string
		inferData string
		validate  func(t *testing.T, feature *transformers.InputFeature, objects *transformers.InputObjects)
	}{
		{
			name:      "simple english text",
			inferData: "hello world",
			validate: func(t *testing.T, feature *transformers.InputFeature, objects *transformers.InputObjects) {
				if feature == nil {
					t.Fatal("feature should not be nil")
				}
				if objects == nil {
					t.Fatal("objects should not be nil")
				}
				if len(feature.TokenIDs) != srv.MaxSeqLength {
					t.Errorf("TokenIDs length = %d, want %d", len(feature.TokenIDs), srv.MaxSeqLength)
				}
				if len(feature.Mask) != srv.MaxSeqLength {
					t.Errorf("Mask length = %d, want %d", len(feature.Mask), srv.MaxSeqLength)
				}
				// First token should be CLS
				if feature.TokenIDs[0] != int32(srv.BertVocab.GetID("[CLS]")) {
					t.Errorf("First token should be CLS, got ID %d", feature.TokenIDs[0])
				}
			},
		},
		{
			name:      "empty string",
			inferData: "",
			validate: func(t *testing.T, feature *transformers.InputFeature, objects *transformers.InputObjects) {
				if feature == nil {
					t.Fatal("feature should not be nil")
				}
				// Should still have CLS and SEP
				if feature.TokenIDs[0] != int32(srv.BertVocab.GetID("[CLS]")) {
					t.Errorf("First token should be CLS")
				}
			},
		},
		{
			name:      "long text that needs truncation",
			inferData: "this is a very long text that should be truncated to fit within the max sequence length limit",
			validate: func(t *testing.T, feature *transformers.InputFeature, objects *transformers.InputObjects) {
				if feature == nil {
					t.Fatal("feature should not be nil")
				}
				if len(feature.TokenIDs) != srv.MaxSeqLength {
					t.Errorf("TokenIDs length should be MaxSeqLength")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			feature, objects := srv.ExportBertGetBertInputFeature(tt.inferData)
			tt.validate(t, feature, objects)
		})
	}
}

func TestBertModelService_GenerateHTTPRequest(t *testing.T) {
	srv := newTestBertModelService(t)

	// Create sample infer inputs
	inferInputs := []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor{
		{Name: "segment_ids", Datatype: "INT32"},
		{Name: "input_ids", Datatype: "INT32"},
		{Name: "input_mask", Datatype: "INT32"},
	}
	inferOutputs := []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor{
		{Name: "output"},
	}

	tests := []struct {
		name      string
		inferData []string
		wantError bool
		validate  func(t *testing.T, body []byte, objects []*transformers.InputObjects, err error)
	}{
		{
			name:      "single input",
			inferData: []string{"hello world"},
			wantError: false,
			validate: func(t *testing.T, body []byte, objects []*transformers.InputObjects, err error) {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if body == nil {
					t.Fatal("body should not be nil")
				}
				if len(objects) != 1 {
					t.Errorf("objects length = %d, want 1", len(objects))
				}
			},
		},
		{
			name:      "batch input",
			inferData: []string{"hello", "world"},
			wantError: false,
			validate: func(t *testing.T, body []byte, objects []*transformers.InputObjects, err error) {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if body == nil {
					t.Fatal("body should not be nil")
				}
				if len(objects) != 2 {
					t.Errorf("objects length = %d, want 2", len(objects))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, objects, err := srv.ExportBertGenerateHTTPRequest(tt.inferData, inferInputs, inferOutputs)
			tt.validate(t, body, objects, err)
		})
	}
}

func TestBertModelService_GenerateGRPCRequest(t *testing.T) {
	srv := newTestBertModelService(t)

	// Create sample infer inputs
	inferInputs := []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor{
		{Name: "segment_ids", Datatype: "INT32"},
		{Name: "input_ids", Datatype: "INT32"},
		{Name: "input_mask", Datatype: "INT32"},
	}

	tests := []struct {
		name      string
		inferData []string
		validate  func(t *testing.T, bytes [][]byte, objects []*transformers.InputObjects)
	}{
		{
			name:      "single input",
			inferData: []string{"hello world"},
			validate: func(t *testing.T, bytes [][]byte, objects []*transformers.InputObjects) {
				if len(bytes) != 3 {
					t.Errorf("bytes length = %d, want 3", len(bytes))
				}
				if len(objects) != 1 {
					t.Errorf("objects length = %d, want 1", len(objects))
				}
				// Each byte slice should have length MaxSeqLength * 4 (INT32 = 4 bytes)
				expectedLen := srv.MaxSeqLength * 4
				for i, b := range bytes {
					if len(b) != expectedLen {
						t.Errorf("bytes[%d] length = %d, want %d", i, len(b), expectedLen)
					}
				}
			},
		},
		{
			name:      "batch input",
			inferData: []string{"hello", "world"},
			validate: func(t *testing.T, bytes [][]byte, objects []*transformers.InputObjects) {
				if len(bytes) != 3 {
					t.Errorf("bytes length = %d, want 3", len(bytes))
				}
				if len(objects) != 2 {
					t.Errorf("objects length = %d, want 2", len(objects))
				}
				// For batch size 2, each byte slice should have length 2 * MaxSeqLength * 4
				expectedLen := 2 * srv.MaxSeqLength * 4
				for i, b := range bytes {
					if len(b) != expectedLen {
						t.Errorf("bytes[%d] length = %d, want %d", i, len(b), expectedLen)
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bytes, objects := srv.ExportBertGenerateGRPCRequest(tt.inferData, inferInputs)
			tt.validate(t, bytes, objects)
		})
	}
}

func TestW2NerModelService_GenerateInitDistInputs(t *testing.T) {
	tests := []struct {
		name     string
		size     int
		validate func(t *testing.T, matrix [][]int32)
	}{
		{
			name: "size 3",
			size: 3,
			validate: func(t *testing.T, matrix [][]int32) {
				if len(matrix) != 3 {
					t.Errorf("matrix rows = %d, want 3", len(matrix))
				}
				for i := range matrix {
					if len(matrix[i]) != 3 {
						t.Errorf("matrix[%d] cols = %d, want 3", i, len(matrix[i]))
					}
				}
				// Check diagonal is 0
				for i := 0; i < 3; i++ {
					if matrix[i][i] != 0 {
						t.Errorf("matrix[%d][%d] = %d, want 0", i, i, matrix[i][i])
					}
				}
				// Check symmetry: matrix[i][j] = -matrix[j][i]
				for i := 0; i < 3; i++ {
					for j := 0; j < 3; j++ {
						if matrix[i][j] != -matrix[j][i] {
							t.Errorf("matrix[%d][%d] = %d, want %d", i, j, matrix[i][j], -matrix[j][i])
						}
					}
				}
			},
		},
		{
			name: "size 5",
			size: 5,
			validate: func(t *testing.T, matrix [][]int32) {
				if len(matrix) != 5 {
					t.Errorf("matrix rows = %d, want 5", len(matrix))
				}
				for i := range matrix {
					if len(matrix[i]) != 5 {
						t.Errorf("matrix[%d] cols = %d, want 5", i, len(matrix[i]))
					}
				}
			},
		},
		{
			name: "size 0",
			size: 0,
			validate: func(t *testing.T, matrix [][]int32) {
				if len(matrix) != 0 {
					t.Errorf("matrix rows = %d, want 0", len(matrix))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matrix := transformers.ExportW2NerGenerateInitDistInputs(tt.size)
			tt.validate(t, matrix)
		})
	}
}

func TestW2NerModelService_GenerateAllTrueSlice(t *testing.T) {
	tests := []struct {
		name     string
		size     int
		validate func(t *testing.T, slice []bool)
	}{
		{
			name: "size 5",
			size: 5,
			validate: func(t *testing.T, slice []bool) {
				if len(slice) != 5 {
					t.Errorf("slice length = %d, want 5", len(slice))
				}
				for i, v := range slice {
					if !v {
						t.Errorf("slice[%d] = %v, want true", i, v)
					}
				}
			},
		},
		{
			name: "size 10",
			size: 10,
			validate: func(t *testing.T, slice []bool) {
				if len(slice) != 10 {
					t.Errorf("slice length = %d, want 10", len(slice))
				}
				for i, v := range slice {
					if !v {
						t.Errorf("slice[%d] = %v, want true", i, v)
					}
				}
			},
		},
		{
			name: "size 0",
			size: 0,
			validate: func(t *testing.T, slice []bool) {
				if len(slice) != 0 {
					t.Errorf("slice length = %d, want 0", len(slice))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			slice := transformers.ExportW2NerGenerateAllTrueSlice(tt.size)
			tt.validate(t, slice)
		})
	}
}
