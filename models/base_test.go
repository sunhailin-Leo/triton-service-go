package models_test

import (
	"testing"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/models"
	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
)

func TestModelService_SetMaxSeqLength(t *testing.T) {
	srv := &models.ModelService{}

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
			result := srv.SetMaxSeqLength(tt.maxLen)

			if result != srv {
				t.Error("Expected method to return *ModelService for chaining")
			}
			if srv.MaxSeqLength != tt.expected {
				t.Errorf("Expected MaxSeqLength=%d, got %d", tt.expected, srv.MaxSeqLength)
			}
		})
	}
}

func TestModelService_SetChineseTokenize(t *testing.T) {
	srv := &models.ModelService{}

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
			result := srv.SetChineseTokenize(tt.isCharMode)

			if result != srv {
				t.Error("Expected method to return *ModelService for chaining")
			}
			if srv.IsChinese != tt.expectedChinese {
				t.Errorf("Expected IsChinese=%v, got %v", tt.expectedChinese, srv.IsChinese)
			}
			if srv.IsChineseCharMode != tt.expectedCharMode {
				t.Errorf("Expected IsChineseCharMode=%v, got %v", tt.expectedCharMode, srv.IsChineseCharMode)
			}
		})
	}
}

func TestModelService_UnsetChineseTokenize(t *testing.T) {
	srv := &models.ModelService{}

	srv.SetChineseTokenize(true)
	result := srv.UnsetChineseTokenize()

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}
	if srv.IsChinese != false {
		t.Errorf("Expected IsChinese=false, got %v", srv.IsChinese)
	}
	if srv.IsChineseCharMode != false {
		t.Errorf("Expected IsChineseCharMode=false, got %v", srv.IsChineseCharMode)
	}
}

func TestModelService_SetModelInferWithGRPC(t *testing.T) {
	srv := &models.ModelService{}

	result := srv.SetModelInferWithGRPC()

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}
	if srv.IsGRPC != true {
		t.Errorf("Expected IsGRPC=true, got %v", srv.IsGRPC)
	}
}

func TestModelService_UnsetModelInferWithGRPC(t *testing.T) {
	srv := &models.ModelService{}

	srv.SetModelInferWithGRPC()
	result := srv.UnsetModelInferWithGRPC()

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}
	if srv.IsGRPC != false {
		t.Errorf("Expected IsGRPC=false, got %v", srv.IsGRPC)
	}
}

func TestModelService_GetModelInferIsGRPC(t *testing.T) {
	srv := &models.ModelService{}

	if srv.GetModelInferIsGRPC() != false {
		t.Error("Expected default GetModelInferIsGRPC()=false")
	}

	srv.SetModelInferWithGRPC()
	if srv.GetModelInferIsGRPC() != true {
		t.Error("Expected GetModelInferIsGRPC()=true after setting")
	}

	srv.UnsetModelInferWithGRPC()
	if srv.GetModelInferIsGRPC() != false {
		t.Error("Expected GetModelInferIsGRPC()=false after unsetting")
	}
}

func TestModelService_GetTokenizerIsChineseMode(t *testing.T) {
	srv := &models.ModelService{}

	if srv.GetTokenizerIsChineseMode() != false {
		t.Error("Expected default GetTokenizerIsChineseMode()=false")
	}

	srv.SetChineseTokenize(true)
	if srv.GetTokenizerIsChineseMode() != true {
		t.Error("Expected GetTokenizerIsChineseMode()=true after setting")
	}

	srv.UnsetChineseTokenize()
	if srv.GetTokenizerIsChineseMode() != false {
		t.Error("Expected GetTokenizerIsChineseMode()=false after unsetting")
	}
}

func TestModelService_SetTokenizerReturnPosInfo(t *testing.T) {
	srv := &models.ModelService{}

	result := srv.SetTokenizerReturnPosInfo()

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}
	if srv.IsReturnPosArray != true {
		t.Errorf("Expected IsReturnPosArray=true, got %v", srv.IsReturnPosArray)
	}
}

func TestModelService_UnsetTokenizerReturnPosInfo(t *testing.T) {
	srv := &models.ModelService{}

	srv.SetTokenizerReturnPosInfo()
	result := srv.UnsetTokenizerReturnPosInfo()

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}
	if srv.IsReturnPosArray != false {
		t.Errorf("Expected IsReturnPosArray=false, got %v", srv.IsReturnPosArray)
	}
}

func TestModelService_SetModelName(t *testing.T) {
	srv := &models.ModelService{}

	result := srv.SetModelName("bert", "base-chinese")

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}

	expected := "bert-base-chinese"
	if srv.ModelName != expected {
		t.Errorf("Expected ModelName=%s, got %s", expected, srv.ModelName)
	}
}

func TestModelService_SetModelNameWithoutDash(t *testing.T) {
	srv := &models.ModelService{}

	result := srv.SetModelNameWithoutDash("bert_base_chinese")

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}

	expected := "bert_base_chinese"
	if srv.ModelName != expected {
		t.Errorf("Expected ModelName=%s, got %s", expected, srv.ModelName)
	}
}

func TestModelService_GetModelName(t *testing.T) {
	srv := &models.ModelService{}

	if srv.GetModelName() != "" {
		t.Error("Expected default GetModelName()=empty string")
	}

	srv.SetModelName("bert", "base-chinese")
	expected := "bert-base-chinese"
	if srv.GetModelName() != expected {
		t.Errorf("Expected GetModelName()=%s, got %s", expected, srv.GetModelName())
	}

	srv.SetModelNameWithoutDash("bert_base_chinese")
	expected = "bert_base_chinese"
	if srv.GetModelName() != expected {
		t.Errorf("Expected GetModelName()=%s, got %s", expected, srv.GetModelName())
	}
}

func TestModelService_SetSecondaryServerURL(t *testing.T) {
	srv := &models.ModelService{}

	secondaryURL := "127.0.0.1:9002"
	result := srv.SetSecondaryServerURL(secondaryURL)

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}
}

func TestModelService_SetSecondaryServerURL_WithTritonService(t *testing.T) {
	tritonSrv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	srv := &models.ModelService{
		TritonService: tritonSrv,
	}

	secondaryURL := "127.0.0.1:9002"

	// Verify SetSecondaryServerURL doesn't panic
	srv.SetSecondaryServerURL(secondaryURL)
}

func TestModelService_SetAPIRequestTimeout(t *testing.T) {
	srv := &models.ModelService{}

	timeout := 30 * time.Second
	result := srv.SetAPIRequestTimeout(timeout)

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}
}

func TestModelService_SetAPIRequestTimeout_WithTritonService(t *testing.T) {
	tritonSrv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	srv := &models.ModelService{
		TritonService: tritonSrv,
	}

	timeout := 30 * time.Second

	// Verify SetAPIRequestTimeout doesn't panic
	srv.SetAPIRequestTimeout(timeout)
}

func TestModelService_SetJsonEncoder(t *testing.T) {
	tritonSrv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	srv := &models.ModelService{
		TritonService: tritonSrv,
	}

	customEncoder := func(v interface{}) ([]byte, error) {
		return []byte(`{"custom":true}`), nil
	}

	result := srv.SetJsonEncoder(customEncoder)

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}

	data, _ := tritonSrv.JsonMarshal(nil)
	if string(data) != `{"custom":true}` {
		t.Errorf("Expected custom encoder to be set, got %s", string(data))
	}
}

func TestModelService_SetJsonDecoder(t *testing.T) {
	tritonSrv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	srv := &models.ModelService{
		TritonService: tritonSrv,
	}

	customDecoder := func(data []byte, v interface{}) error {
		if m, ok := v.(*map[string]string); ok {
			(*m)["custom"] = "decoded"
		}
		return nil
	}

	result := srv.SetJsonDecoder(customDecoder)

	if result != srv {
		t.Error("Expected method to return *ModelService for chaining")
	}

	decoded := make(map[string]string)
	if err := tritonSrv.JsonUnmarshal([]byte(`{}`), &decoded); err != nil {
		t.Fatalf("JsonUnmarshal failed: %v", err)
	}
	if decoded["custom"] != "decoded" {
		t.Errorf("Expected custom decoder to be set, got %s", decoded["custom"])
	}
}
