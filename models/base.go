package models

import (
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

// GenerateModelInferRequest model input callback.
type GenerateModelInferRequest func() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor

// GenerateModelInferOutputRequest model output callback.
type GenerateModelInferOutputRequest func(params ...interface{}) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor

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

// SetJsonEncoder set json encoder
func (m *ModelService) SetJsonEncoder(encoder utils.JSONMarshal) *ModelService {
	m.TritonService.SetJSONEncoder(encoder)
	return m
}

// SetJsonDecoder set json decoder
func (m *ModelService) SetJsonDecoder(decoder utils.JSONUnmarshal) *ModelService {
	m.TritonService.SetJsonDecoder(decoder)
	return m
}

////////////////////////////////////////////////// Flag Switch API //////////////////////////////////////////////////

//////////////////////////////////////////// Triton Service API Function ////////////////////////////////////////////

// CheckServerReady check server is ready.
func (m *ModelService) CheckServerReady() (bool, error) {
	return m.TritonService.CheckServerReady()
}

// CheckServerAlive check server is alive.
func (m *ModelService) CheckServerAlive() (bool, error) {
	return m.TritonService.CheckServerAlive()
}

// CheckModelReady check model is ready.
func (m *ModelService) CheckModelReady(modelName, modelVersion string) (bool, error) {
	return m.TritonService.CheckModelReady(modelName, modelVersion)
}

// GetServerMeta get server meta.
func (m *ModelService) GetServerMeta() (*nvidia_inferenceserver.ServerMetadataResponse, error) {
	return m.TritonService.ServerMetadata()
}

// GetModelMeta get model meta.
func (m *ModelService) GetModelMeta(modelName, modelVersion string) (*nvidia_inferenceserver.ModelMetadataResponse, error) {
	return m.TritonService.ModelMetadataRequest(modelName, modelVersion)
}

// GetAllModelInfo get all model info.
func (m *ModelService) GetAllModelInfo(repoName string, isReady bool) (*nvidia_inferenceserver.RepositoryIndexResponse, error) {
	return m.TritonService.ModelIndex(repoName, isReady)
}

// GetModelConfig get model config.
func (m *ModelService) GetModelConfig(modelName, modelVersion string) (interface{}, error) {
	return m.TritonService.ModelConfiguration(modelName, modelVersion)
}

// GetModelInferStats get model infer stats.
func (m *ModelService) GetModelInferStats(modelName, modelVersion string) (*nvidia_inferenceserver.ModelStatisticsResponse, error) {
	return m.TritonService.ModelInferStats(modelName, modelVersion)
}

//////////////////////////////////////////// Triton Service API Function ////////////////////////////////////////////
