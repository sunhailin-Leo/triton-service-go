package models

import (
	"context"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

// GenerateModelInferRequest model input callback.
type GenerateModelInferRequest func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor

// GenerateModelInferOutputRequest model output callback.
type GenerateModelInferOutputRequest func(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor

// ModelService is the base service for all model implementations.
//
// Deprecated field access: All exported fields are retained for backward compatibility.
// New code should use the provided setter/getter methods (e.g., SetMaxSeqLength, SetModelInferWithGRPC)
// or functional options (e.g., WithBertMaxSeqLength) instead of accessing fields directly.
// Direct field mutation is not concurrency-safe.
type ModelService struct {
	IsGRPC                          bool
	IsChinese                       bool
	IsChineseCharMode               bool
	IsReturnPosArray                bool
	MaxSeqLength                    int
	ModelName                       string
	TritonService                   *nvidia_inferenceserver.TritonClientService
	InferCallback                   nvidia_inferenceserver.DecoderFunc
	GenerateModelInferRequest       GenerateModelInferRequest
	GenerateModelInferOutputRequest GenerateModelInferOutputRequest
}

////////////////////////////////////////////////// Flag Switch API //////////////////////////////////////////////////

// SetMaxSeqLength Set model infer max sequence length.
func (m *ModelService) SetMaxSeqLength(maxSeqLen int) *ModelService {
	m.MaxSeqLength = maxSeqLen

	return m
}

// SetChineseTokenize Use Chinese Tokenize when tokenize infer data.
func (m *ModelService) SetChineseTokenize(isCharMode bool) *ModelService {
	m.IsChinese = true
	m.IsChineseCharMode = isCharMode

	return m
}

// UnsetChineseTokenize Un-use Chinese Tokenize when tokenize infer data.
func (m *ModelService) UnsetChineseTokenize() *ModelService {
	m.IsChinese = false
	m.IsChineseCharMode = false

	return m
}

// SetModelInferWithGRPC Use grpc to call triton.
func (m *ModelService) SetModelInferWithGRPC() *ModelService {
	m.IsGRPC = true

	return m
}

// UnsetModelInferWithGRPC Un-use grpc to call triton.
func (m *ModelService) UnsetModelInferWithGRPC() *ModelService {
	m.IsGRPC = false

	return m
}

// GetModelInferIsGRPC Get isGRPC flag.
func (m *ModelService) GetModelInferIsGRPC() bool {
	return m.IsGRPC
}

// GetTokenizerIsChineseMode Get isChinese flag.
func (m *ModelService) GetTokenizerIsChineseMode() bool {
	return m.IsChinese
}

// SetTokenizerReturnPosInfo Set tokenizer return pos info.
func (m *ModelService) SetTokenizerReturnPosInfo() *ModelService {
	m.IsReturnPosArray = true

	return m
}

// UnsetTokenizerReturnPosInfo Un-set tokenizer return pos info.
func (m *ModelService) UnsetTokenizerReturnPosInfo() *ModelService {
	m.IsReturnPosArray = false

	return m
}

// SetModelName Set model name must equal to Triton config.pbtxt model name.
func (m *ModelService) SetModelName(modelPrefix, modelName string) *ModelService {
	m.ModelName = modelPrefix + "-" + modelName

	return m
}

func (m *ModelService) SetModelNameWithoutDash(modelName string) *ModelService {
	m.ModelName = modelName

	return m
}

// GetModelName Get model name.
func (m *ModelService) GetModelName() string { return m.ModelName }

// SetSecondaryServerURL set secondary server url【Only HTTP】
func (m *ModelService) SetSecondaryServerURL(url string) *ModelService {
	if m.TritonService != nil {
		m.TritonService.SetSecondaryServerURL(url)
	}
	return m
}

// SetAPIRequestTimeout set api request timeout
func (m *ModelService) SetAPIRequestTimeout(timeout time.Duration) *ModelService {
	if m.TritonService != nil {
		m.TritonService.SetAPIRequestTimeout(timeout)
	}
	return m
}

// SetJSONEncoder set json encoder
//
// Deprecated: Use WithJSONEncoder option during client construction instead.
func (m *ModelService) SetJSONEncoder(encoder utils.JSONMarshal) *ModelService {
	m.TritonService.SetJSONEncoder(encoder)
	return m
}

// SetJSONDecoder set json decoder
//
// Deprecated: Use WithJSONDecoder option during client construction instead.
func (m *ModelService) SetJSONDecoder(decoder utils.JSONUnmarshal) *ModelService {
	m.TritonService.SetJSONDecoder(decoder)
	return m
}

////////////////////////////////////////////////// Flag Switch API //////////////////////////////////////////////////

//////////////////////////////////////////// Triton Service API Function ////////////////////////////////////////////

// CheckServerReady check server is ready.
func (m *ModelService) CheckServerReady(ctx context.Context) (bool, error) {
	return m.TritonService.CheckServerReady(ctx)
}

// CheckServerAlive check server is alive.
func (m *ModelService) CheckServerAlive(ctx context.Context) (bool, error) {
	return m.TritonService.CheckServerAlive(ctx)
}

// CheckModelReady check model is ready.
func (m *ModelService) CheckModelReady(ctx context.Context, modelName, modelVersion string) (bool, error) {
	return m.TritonService.CheckModelReady(ctx, modelName, modelVersion)
}

// GetServerMeta get server meta.
func (m *ModelService) GetServerMeta(ctx context.Context) (*nvidia_inferenceserver.ServerMetadataResponse, error) {
	return m.TritonService.ServerMetadata(ctx)
}

// GetModelMeta get model meta.
func (m *ModelService) GetModelMeta(ctx context.Context, modelName, modelVersion string) (*nvidia_inferenceserver.ModelMetadataResponse, error) {
	return m.TritonService.ModelMetadataRequest(ctx, modelName, modelVersion)
}

// GetAllModelInfo get all model info.
func (m *ModelService) GetAllModelInfo(ctx context.Context, repoName string, isReady bool) (*nvidia_inferenceserver.RepositoryIndexResponse, error) {
	return m.TritonService.ModelIndex(ctx, repoName, isReady)
}

// GetModelConfig get model config.
func (m *ModelService) GetModelConfig(ctx context.Context, modelName, modelVersion string) (*nvidia_inferenceserver.ModelConfigResponse, error) {
	return m.TritonService.ModelConfiguration(ctx, modelName, modelVersion)
}

// GetModelInferStats get model infer stats.
func (m *ModelService) GetModelInferStats(ctx context.Context, modelName, modelVersion string) (*nvidia_inferenceserver.ModelStatisticsResponse, error) {
	return m.TritonService.ModelInferStats(ctx, modelName, modelVersion)
}

//////////////////////////////////////////// Triton Service API Function ////////////////////////////////////////////
