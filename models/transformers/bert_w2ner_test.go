package transformers_test

import (
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
)

func testGenerateW2NERModelInferRequest() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor {
	return []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor{
		{Name: "input_ids", Datatype: "INT32"},
		{Name: "grid_mask2d", Datatype: "BOOL"},
		{Name: "dist_inputs", Datatype: "INT32"},
		{Name: "pieces2word", Datatype: "BOOL"},
	}
}

func testGenerateW2NERModelInferOutputRequest(params ...interface{}) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor {
	return []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor{
		{Name: "logic"},
	}
}

func testW2NERModelInferCallback(inferResponse interface{}, params ...interface{}) ([]interface{}, error) {
	return nil, nil
}

func TestNewW2NERModelService_Success(t *testing.T) {
	srv, err := transformers.NewW2NERModelService(
		"../../test/bert-chinese-vocab.txt",
		"127.0.0.1:9001",
		nil,
		nil,
		testGenerateW2NERModelInferRequest,
		testGenerateW2NERModelInferOutputRequest,
		testW2NERModelInferCallback,
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
	if srv.BertModelService == nil {
		t.Error("Expected BertModelService to be initialized")
	}
}

func TestNewW2NERModelService_NilCallbacks(t *testing.T) {
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
			outputCallback: testGenerateW2NERModelInferOutputRequest,
			inferCallback:  testW2NERModelInferCallback,
			expectError:    true,
		},
		{
			name:           "nil output callback",
			inputCallback:  testGenerateW2NERModelInferRequest,
			outputCallback: nil,
			inferCallback:  testW2NERModelInferCallback,
			expectError:    true,
		},
		{
			name:           "nil infer callback",
			inputCallback:  testGenerateW2NERModelInferRequest,
			outputCallback: testGenerateW2NERModelInferOutputRequest,
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
			_, err := transformers.NewW2NERModelService(
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

func TestNewW2NERModelService_InvalidVocabPath(t *testing.T) {
	_, err := transformers.NewW2NERModelService(
		"../../test/non-existent-vocab.txt",
		"127.0.0.1:9001",
		nil,
		nil,
		testGenerateW2NERModelInferRequest,
		testGenerateW2NERModelInferOutputRequest,
		testW2NERModelInferCallback,
	)

	if err == nil {
		t.Error("Expected error for invalid vocab path, got nil")
	}
}
