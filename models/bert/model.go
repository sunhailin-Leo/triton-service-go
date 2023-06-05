package bert

import (
	"encoding/binary"
	"strings"
	"time"

	"github.com/goccy/go-json"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"

	"github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"
	"github.com/sunhailin-Leo/triton-service-go/utils"
)

const (
	DefaultMaxSeqLength                      int    = 48
	ModelRespBodyOutputBinaryDataKey         string = "binary_data"
	ModelRespBodyOutputClassificationDataKey string = "classification"
	ModelBertModelSegmentIdsKey              string = "segment_ids"
	ModelBertModelInputIdsKey                string = "input_ids"
	ModelBertModelInputMaskKey               string = "input_mask"
	ModelInt32DataType                       string = "INT32"
	ModelInt64DataType                       string = "INT64"
)

type ModelService struct {
	// isTraceDuration                 bool
	isGRPC                          bool
	isChinese                       bool
	isReturnPosArray                bool
	maxSeqLength                    int
	modelName                       string
	tritonService                   *nvidia_inferenceserver.TritonClientService
	inferCallback                   nvidia_inferenceserver.DecoderFunc
	BertVocab                       Dict
	BertTokenizer                   *WordPieceTokenizer
	generateModelInferRequest       GenerateModelInferRequest
	generateModelInferOutputRequest GenerateModelInferOutputRequest
}

////////////////////////////////////////////////// Flag Switch API //////////////////////////////////////////////////

//// SetModelInferWithTrace Set model infer trace obj.
// func (m *ModelService) SetModelInferWithTrace() *ModelService {
//	 m.isTraceDuration = true
//	 return m
// }
//
//// UnsetModelInferWithTrace unset model infer trace obj.
// func (m *ModelService) UnsetModelInferWithTrace() *ModelService {
// 	m.isTraceDuration = false
// 	return m
// }

// SetMaxSeqLength Set model infer max sequence length.
func (m *ModelService) SetMaxSeqLength(maxSeqLen int) *ModelService {
	m.maxSeqLength = maxSeqLen

	return m
}

// SetChineseTokenize Use Chinese Tokenize when tokenize infer data.
func (m *ModelService) SetChineseTokenize() *ModelService {
	m.isChinese = true

	return m
}

// UnsetChineseTokenize Un-use Chinese Tokenize when tokenize infer data.
func (m *ModelService) UnsetChineseTokenize() *ModelService {
	m.isChinese = false

	return m
}

// SetModelInferWithGRPC Use grpc to call triton.
func (m *ModelService) SetModelInferWithGRPC() *ModelService {
	m.isGRPC = true

	return m
}

// UnsetModelInferWithGRPC Un-use grpc to call triton.
func (m *ModelService) UnsetModelInferWithGRPC() *ModelService {
	m.isGRPC = false

	return m
}

// GetModelInferIsGRPC Get isGRPC flag.
func (m *ModelService) GetModelInferIsGRPC() bool {
	return m.isGRPC
}

// GetTokenizerIsChineseMode Get isChinese flag.
func (m *ModelService) GetTokenizerIsChineseMode() bool {
	return m.isChinese
}

// SetTokenizerReturnPosInfo Set tokenizer return pos info.
func (m *ModelService) SetTokenizerReturnPosInfo() *ModelService {
	m.isReturnPosArray = true

	return m
}

// UnsetTokenizerReturnPosInfo Un-set tokenizer return pos info.
func (m *ModelService) UnsetTokenizerReturnPosInfo() *ModelService {
	m.isReturnPosArray = false

	return m
}

// SetModelName Set model name must equal to Triton config.pbtxt model name.
func (m *ModelService) SetModelName(modelPrefix, modelName string) *ModelService {
	// TODO maybe not use dash to separate modelPrefix and modelName
	m.modelName = modelPrefix + "-" + modelName

	return m
}

// GetModelName Get model name.
func (m *ModelService) GetModelName() string { return m.modelName }

// SetSecondaryServerURL set secondary server url【Only HTTP】
func (m *ModelService) SetSecondaryServerURL(url string) *ModelService {
	if m.tritonService != nil {
		m.tritonService.SetSecondaryServerURL(url)
	}
	return m
}

////////////////////////////////////////////////// Flag Switch API //////////////////////////////////////////////////

///////////////////////////////////////// Bert Service Pre-Process Function /////////////////////////////////////////

// getTokenizerResult Get Tokenizer result from different tokenizers.
func (m *ModelService) getTokenizerResult(inferData string) []string {
	if m.isChinese {
		return GetStrings(m.BertTokenizer.TokenizeChinese(strings.ToLower(inferData)))
	}
	return GetStrings(m.BertTokenizer.Tokenize(inferData))
}

// getTokenizerResultWithOffsets Get Tokenizer result from different tokenizers with offsets.
func (m *ModelService) getTokenizerResultWithOffsets(inferData string) ([]string, []OffsetsType) {
	if m.isChinese {
		tokenizerResult := m.BertTokenizer.TokenizeChinese(strings.ToLower(inferData))

		return GetStrings(tokenizerResult), GetOffsets(tokenizerResult)
	}
	tokenizerResult := m.BertTokenizer.Tokenize(inferData)

	return GetStrings(tokenizerResult), GetOffsets(tokenizerResult)
}

// getBertInputFeature Get Bert Feature (before Make HTTP or GRPC Request).
func (m *ModelService) getBertInputFeature(inferData string) (*InputFeature, *InputObjects) {
	// Replace BertDataSplitString Here, so the parts is 1, no need to use strings.Split and decrease a for-loop.
	if strings.Index(inferData, DataSplitString) > 0 {
		inferData = strings.ReplaceAll(inferData, DataSplitString, "")
	}
	// InputFeature
	// feature.TypeIDs  == segment_ids
	// feature.TokenIDs == input_ids
	// feature.Mask     == input_mask
	feature := &InputFeature{
		Tokens:   make([]string, m.maxSeqLength),
		TokenIDs: make([]int32, m.maxSeqLength),
		Mask:     make([]int32, m.maxSeqLength),
		TypeIDs:  make([]int32, m.maxSeqLength),
	}
	inputObjects := &InputObjects{Input: inferData}
	// inferData only a short text, so it`s length always 1.
	// truncate w/ space for CLS/SEP, 1 for sequence length and 1 for the last index
	sequence := make([][]string, 1)
	if m.isReturnPosArray {
		sequence[0], inputObjects.PosArray = m.getTokenizerResultWithOffsets(inferData)
	} else {
		sequence[0] = m.getTokenizerResult(inferData)
	}
	sequence = utils.StringSliceTruncate(sequence, m.maxSeqLength-2)
	for i := 0; i <= len(sequence[0])+1; i++ {
		feature.Mask[i] = 1
		switch {
		case i == 0:
			feature.TokenIDs[i] = int32(m.BertVocab.GetID(DefaultCLS))
			feature.Tokens[i] = DefaultCLS
		case i == len(sequence[0])+1:
			feature.TokenIDs[i] = int32(m.BertVocab.GetID(DefaultSEP))
			feature.Tokens[i] = DefaultSEP
		default:
			feature.TokenIDs[i] = int32(m.BertVocab.GetID(sequence[0][i-1]))
			feature.Tokens[i] = sequence[0][i-1]
		}
	}
	inputObjects.Tokens = feature.Tokens
	return feature, inputObjects
}

// generateHTTPOutputs For HTTP Output.
func (m *ModelService) generateHTTPOutputs(
	inferOutputs []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor,
) []HTTPOutput {
	requestOutputs := make([]HTTPOutput, len(inferOutputs))
	for i := range inferOutputs {
		requestOutputs[i] = HTTPOutput{Name: inferOutputs[i].Name}
		if _, ok := inferOutputs[i].Parameters[ModelRespBodyOutputBinaryDataKey]; ok {
			requestOutputs[i].Parameters.BinaryData =
				inferOutputs[i].Parameters[ModelRespBodyOutputBinaryDataKey].GetBoolParam()
		}
		if _, ok := inferOutputs[i].Parameters[ModelRespBodyOutputClassificationDataKey]; ok {
			requestOutputs[i].Parameters.Classification =
				inferOutputs[i].Parameters[ModelRespBodyOutputClassificationDataKey].GetInt64Param()
		}
	}
	return requestOutputs
}

// generateHTTPInputs get bert input feature for http request
// inferDataArr: model infer data slice
// inferInputs: triton inference server input tensor.
func (m *ModelService) generateHTTPInputs(
	inferDataArr []string, inferInputs []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor,
) ([]HTTPBatchInput, []*InputObjects) {
	// Bert Feature
	batchModelInputObjs := make([]*InputObjects, len(inferDataArr))
	batchRequestInputs := make([]HTTPBatchInput, len(inferInputs))

	inferDataObjs := make([][][]int32, len(inferDataArr))
	for i := range inferDataArr {
		feature, inputObject := m.getBertInputFeature(inferDataArr[i])
		batchModelInputObjs[i] = inputObject
		inferDataObjs[i] = [][]int32{feature.TypeIDs, feature.TokenIDs, feature.Mask}
	}
	inferDataObjs = utils.SliceTransposeFor3D(inferDataObjs)
	for i := range inferInputs {
		batchRequestInputs[i] = HTTPBatchInput{
			Name:     inferInputs[i].Name,
			Shape:    inferInputs[i].Shape,
			DataType: inferInputs[i].Datatype,
			Data:     inferDataObjs[i],
		}
	}
	return batchRequestInputs, batchModelInputObjs
}

// generateHTTPRequest HTTP Request Data Generate.
func (m *ModelService) generateHTTPRequest(
	inferDataArr []string,
	inferInputs []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor,
	inferOutputs []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor,
) ([]byte, []*InputObjects, error) {
	// Generate batch request json body
	requestInputBody, modelInputObj := m.generateHTTPInputs(inferDataArr, inferInputs)
	jsonBody, jsonEncodeErr := json.Marshal(&HTTPRequestBody{
		Inputs:  requestInputBody,
		Outputs: m.generateHTTPOutputs(inferOutputs),
	})
	if jsonEncodeErr != nil {
		return nil, nil, jsonEncodeErr
	}
	return jsonBody, modelInputObj, nil
}

// grpcInt32SliceToLittleEndianByteSlice int32 slice to byte slice with little endian.
func (m *ModelService) grpcInt32SliceToLittleEndianByteSlice(maxLen int, input []int32, inputType string) []byte {
	switch inputType {
	case ModelInt32DataType:
		var returnByte []byte
		bs := make([]byte, 4)
		for i := 0; i < maxLen; i++ {
			binary.LittleEndian.PutUint32(bs, uint32(input[i]))
			returnByte = append(returnByte, bs...)
		}
		return returnByte
	case ModelInt64DataType:
		var returnByte []byte
		bs := make([]byte, 8)
		for i := 0; i < maxLen; i++ {
			binary.LittleEndian.PutUint64(bs, uint64(input[i]))
			returnByte = append(returnByte, bs...)
		}
		return returnByte
	default:
		return nil
	}
}

// generateGRPCRequest GRPC Request Data Generate
func (m *ModelService) generateGRPCRequest(
	inferDataArr []string,
	inferInputTensor []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor,
) ([][]byte, []*InputObjects) {
	// size is: len(inferDataArr) * m.maxSeqLength * 4
	var segmentIdsBytes, inputIdsBytes, inputMaskBytes []byte
	batchModelInputObjs := make([]*InputObjects, len(inferDataArr))
	for i := range inferDataArr {
		feature, inputObject := m.getBertInputFeature(inferDataArr[i])
		// feature.TypeIDs  == segment_ids
		// feature.TokenIDs == input_ids
		// feature.Mask     == input_mask
		// Temp variable to hold out converted int32 -> []byte
		for j := range inferInputTensor {
			switch inferInputTensor[j].Name {
			case ModelBertModelSegmentIdsKey:
				segmentIdsBytes = append(
					segmentIdsBytes,
					m.grpcInt32SliceToLittleEndianByteSlice(
						m.maxSeqLength, feature.TypeIDs, inferInputTensor[j].Datatype)...,
				)
			case ModelBertModelInputIdsKey:
				inputIdsBytes = append(
					inputIdsBytes,
					m.grpcInt32SliceToLittleEndianByteSlice(
						m.maxSeqLength, feature.TokenIDs, inferInputTensor[j].Datatype)...,
				)
			case ModelBertModelInputMaskKey:
				inputMaskBytes = append(
					inputMaskBytes,
					m.grpcInt32SliceToLittleEndianByteSlice(
						m.maxSeqLength, feature.Mask, inferInputTensor[j].Datatype)...,
				)
			}
		}
		batchModelInputObjs[i] = inputObject
	}
	return [][]byte{segmentIdsBytes, inputIdsBytes, inputMaskBytes}, batchModelInputObjs
}

///////////////////////////////////////// Bert Service Pre-Process Function /////////////////////////////////////////

//////////////////////////////////////////// Triton Service API Function ////////////////////////////////////////////

// CheckServerReady check server is ready.
func (m *ModelService) CheckServerReady(requestTimeout time.Duration) (bool, error) {
	return m.tritonService.CheckServerReady(requestTimeout)
}

// CheckServerAlive check server is alive.
func (m *ModelService) CheckServerAlive(requestTimeout time.Duration) (bool, error) {
	return m.tritonService.CheckServerAlive(requestTimeout)
}

// CheckModelReady check model is ready.
func (m *ModelService) CheckModelReady(
	modelName, modelVersion string, requestTimeout time.Duration,
) (bool, error) {
	return m.tritonService.CheckModelReady(modelName, modelVersion, requestTimeout)
}

// GetServerMeta get server meta.
func (m *ModelService) GetServerMeta(
	requestTimeout time.Duration,
) (*nvidia_inferenceserver.ServerMetadataResponse, error) {
	return m.tritonService.ServerMetadata(requestTimeout)
}

// GetModelMeta get model meta.
func (m *ModelService) GetModelMeta(
	modelName, modelVersion string, requestTimeout time.Duration,
) (*nvidia_inferenceserver.ModelMetadataResponse, error) {
	return m.tritonService.ModelMetadataRequest(modelName, modelVersion, requestTimeout)
}

// GetAllModelInfo get all model info.
func (m *ModelService) GetAllModelInfo(
	repoName string, isReady bool, requestTimeout time.Duration,
) (*nvidia_inferenceserver.RepositoryIndexResponse, error) {
	return m.tritonService.ModelIndex(repoName, isReady, requestTimeout)
}

// GetModelConfig get model config.
func (m *ModelService) GetModelConfig(
	modelName, modelVersion string, requestTimeout time.Duration,
) (interface{}, error) {
	return m.tritonService.ModelConfiguration(modelName, modelVersion, requestTimeout)
}

// GetModelInferStats get model infer stats.
func (m *ModelService) GetModelInferStats(
	modelName, modelVersion string, requestTimeout time.Duration,
) (*nvidia_inferenceserver.ModelStatisticsResponse, error) {
	return m.tritonService.ModelInferStats(modelName, modelVersion, requestTimeout)
}

// ModelInfer API to call Triton Inference Server.
func (m *ModelService) ModelInfer(
	inferData []string,
	modelName, modelVersion string,
	requestTimeout time.Duration,
	params ...interface{},
) ([]interface{}, error) {
	// Create request input/output tensors
	inferInputs := m.generateModelInferRequest(len(inferData), m.maxSeqLength)
	inferOutputs := m.generateModelInferOutputRequest(params...)
	if m.isGRPC {
		// GRPC Infer
		grpcRawInputs, grpcInputData := m.generateGRPCRequest(inferData, inferInputs)
		if grpcRawInputs == nil {
			return nil, utils.ErrEmptyGRPCRequestBody
		}
		return m.tritonService.ModelGRPCInfer(
			inferInputs, inferOutputs, grpcRawInputs, modelName, modelVersion, requestTimeout,
			m.inferCallback, m, grpcInputData, params,
		)
	}
	httpRequestBody, httpInputData, err := m.generateHTTPRequest(inferData, inferInputs, inferOutputs)
	if err != nil {
		return nil, err
	}
	if httpRequestBody == nil {
		return nil, utils.ErrEmptyHTTPRequestBody
	}
	// HTTP Infer
	return m.tritonService.ModelHTTPInfer(
		httpRequestBody, modelName, modelVersion, requestTimeout,
		m.inferCallback, m, httpInputData,
	)
}

//////////////////////////////////////////// Triton Service API Function ////////////////////////////////////////////

func NewModelService(
	bertVocabPath, httpAddr string,
	httpClient *fasthttp.Client, grpcConn *grpc.ClientConn,
	modelInputCallback GenerateModelInferRequest,
	modelOutputCallback GenerateModelInferOutputRequest,
	modelInferCallback nvidia_inferenceserver.DecoderFunc,
) (*ModelService, error) {
	// 0、callback function validation
	if modelInputCallback == nil || modelOutputCallback == nil || modelInferCallback == nil {
		return nil, utils.ErrEmptyCallbackFunc
	}
	// 1、Init Vocab
	voc, vocabReadErr := VocabFromFile(bertVocabPath)
	if vocabReadErr != nil {
		return nil, vocabReadErr
	}
	// 2、Init Service
	srv := &ModelService{
		maxSeqLength:                    DefaultMaxSeqLength,
		tritonService:                   nvidia_inferenceserver.NewTritonClientForAll(httpAddr, httpClient, grpcConn),
		inferCallback:                   modelInferCallback,
		BertVocab:                       voc,
		BertTokenizer:                   NewWordPieceTokenizer(voc),
		generateModelInferRequest:       modelInputCallback,
		generateModelInferOutputRequest: modelOutputCallback,
	}
	return srv, nil
}
