package transformers

import (
	"encoding/binary"
	"strings"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/models"
	"github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"
	"github.com/sunhailin-Leo/triton-service-go/utils"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
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

type BertModelService struct {
	models.ModelService

	BertVocab     Dict
	BertTokenizer *WordPieceTokenizer
}

///////////////////////////////////////// Bert Service Pre-Process Function /////////////////////////////////////////

// getTokenizerResult Get Tokenizer result from different tokenizers.
func (m *BertModelService) getTokenizerResult(inferData string) []string {
	if m.IsChinese {
		if m.IsChineseCharMode {
			return GetStrings(m.BertTokenizer.TokenizeChineseCharMode(strings.ToLower(inferData)))
		}
		return GetStrings(m.BertTokenizer.TokenizeChinese(strings.ToLower(inferData)))
	}
	return GetStrings(m.BertTokenizer.Tokenize(inferData))
}

// getTokenizerResultWithOffsets Get Tokenizer result from different tokenizers with offsets.
func (m *BertModelService) getTokenizerResultWithOffsets(inferData string) ([]string, []OffsetsType) {
	if m.IsChinese {
		var tokenizerResult []StringOffsetsPair
		if m.IsChineseCharMode {
			tokenizerResult = m.BertTokenizer.TokenizeChineseCharMode(strings.ToLower(inferData))
		} else {
			tokenizerResult = m.BertTokenizer.TokenizeChinese(strings.ToLower(inferData))
		}

		return GetStrings(tokenizerResult), GetOffsets(tokenizerResult)
	}
	tokenizerResult := m.BertTokenizer.Tokenize(inferData)

	return GetStrings(tokenizerResult), GetOffsets(tokenizerResult)
}

// getBertInputFeature Get Bert Feature (before Make HTTP or GRPC Request).
func (m *BertModelService) getBertInputFeature(inferData string) (*InputFeature, *InputObjects) {
	// Replace BertDataSplitString Here, so the parts is 1, no need to use strings.Split and decrease a for-loop.
	if strings.Index(inferData, DataSplitString) > 0 {
		inferData = strings.ReplaceAll(inferData, DataSplitString, "")
	}
	// InputFeature
	// feature.TypeIDs  == segment_ids
	// feature.TokenIDs == input_ids
	// feature.Mask     == input_mask
	feature := &InputFeature{
		Tokens:   make([]string, m.MaxSeqLength),
		TokenIDs: make([]int32, m.MaxSeqLength),
		Mask:     make([]int32, m.MaxSeqLength),
		TypeIDs:  make([]int32, m.MaxSeqLength),
	}
	inputObjects := &InputObjects{Input: inferData}
	// inferData only a short text, so it`s length always 1.
	// truncate w/ space for CLS/SEP, 1 for sequence length and 1 for the last index
	sequence := make([][]string, 1)
	if m.IsReturnPosArray {
		sequence[0], inputObjects.PosArray = m.getTokenizerResultWithOffsets(inferData)
	} else {
		sequence[0] = m.getTokenizerResult(inferData)
	}
	sequence = utils.StringSliceTruncate(sequence, m.MaxSeqLength-2)
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
func (m *BertModelService) generateHTTPOutputs(
	inferOutputs []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor,
) []HTTPOutput {
	requestOutputs := make([]HTTPOutput, len(inferOutputs))
	for i := range inferOutputs {
		requestOutputs[i] = HTTPOutput{Name: inferOutputs[i].Name}
		if len(inferOutputs[i].Parameters) > 0 {
			if _, ok := inferOutputs[i].Parameters[ModelRespBodyOutputBinaryDataKey]; ok {
				requestOutputs[i].Parameters.BinaryData = inferOutputs[i].Parameters[ModelRespBodyOutputBinaryDataKey].
					GetBoolParam()
			}
			if _, ok := inferOutputs[i].Parameters[ModelRespBodyOutputClassificationDataKey]; ok {
				requestOutputs[i].Parameters.Classification = inferOutputs[i].Parameters[ModelRespBodyOutputClassificationDataKey].
					GetInt64Param()
			}
		}
	}
	return requestOutputs
}

// generateHTTPInputs get bert input feature for http request
// inferDataArr: model infer data slice
// inferInputs: triton inference server input tensor.
func (m *BertModelService) generateHTTPInputs(
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
			Shape:    []int64{int64(len(inferDataArr)), int64(m.MaxSeqLength)},
			DataType: inferInputs[i].Datatype,
			Data:     inferDataObjs[i],
		}
	}
	return batchRequestInputs, batchModelInputObjs
}

// generateHTTPRequest HTTP Request Data Generate.
func (m *BertModelService) generateHTTPRequest(
	inferDataArr []string,
	inferInputs []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor,
	inferOutputs []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor,
) ([]byte, []*InputObjects, error) {
	// Generate batch request json body
	requestInputBody, modelInputObj := m.generateHTTPInputs(inferDataArr, inferInputs)
	jsonBody, jsonEncodeErr := m.TritonService.JsonMarshal(&HTTPRequestBody{
		Inputs:  requestInputBody,
		Outputs: m.generateHTTPOutputs(inferOutputs),
	})
	if jsonEncodeErr != nil {
		return nil, nil, jsonEncodeErr
	}
	return jsonBody, modelInputObj, nil
}

// grpcInt32SliceToLittleEndianByteSlice int32 slice to byte slice with little endian.
func (m *BertModelService) grpcInt32SliceToLittleEndianByteSlice(maxLen int, input []int32, inputType string) []byte {
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
func (m *BertModelService) generateGRPCRequest(
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
						m.MaxSeqLength, feature.TypeIDs, inferInputTensor[j].Datatype)...,
				)
			case ModelBertModelInputIdsKey:
				inputIdsBytes = append(
					inputIdsBytes,
					m.grpcInt32SliceToLittleEndianByteSlice(
						m.MaxSeqLength, feature.TokenIDs, inferInputTensor[j].Datatype)...,
				)
			case ModelBertModelInputMaskKey:
				inputMaskBytes = append(
					inputMaskBytes,
					m.grpcInt32SliceToLittleEndianByteSlice(
						m.MaxSeqLength, feature.Mask, inferInputTensor[j].Datatype)...,
				)
			}
		}
		batchModelInputObjs[i] = inputObject
	}
	return [][]byte{segmentIdsBytes, inputIdsBytes, inputMaskBytes}, batchModelInputObjs
}

///////////////////////////////////////// Bert Service Pre-Process Function /////////////////////////////////////////

//////////////////////////////////////////// Triton Service API Function ////////////////////////////////////////////

// ModelInfer API to call Triton Inference Server.
func (m *BertModelService) ModelInfer(
	inferData []string,
	modelName, modelVersion string,
	requestTimeout time.Duration,
	params ...interface{},
) ([]interface{}, error) {
	// Create request input/output tensors
	inferInputs := m.GenerateModelInferRequest()
	inferOutputs := m.GenerateModelInferOutputRequest(params...)
	if m.IsGRPC {
		// GRPC Infer
		grpcRawInputs, grpcInputData := m.generateGRPCRequest(inferData, inferInputs)
		if grpcRawInputs == nil {
			return nil, utils.ErrEmptyGRPCRequestBody
		}
		return m.TritonService.ModelGRPCInfer(
			inferInputs, inferOutputs, grpcRawInputs, modelName, modelVersion, requestTimeout,
			m.InferCallback, m, grpcInputData, params,
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
	return m.TritonService.ModelHTTPInfer(
		httpRequestBody, modelName, modelVersion, requestTimeout,
		m.InferCallback, m, httpInputData, params,
	)
}

//////////////////////////////////////////// Triton Service API Function ////////////////////////////////////////////

func NewBertModelService(
	bertVocabPath, httpAddr string,
	httpClient *fasthttp.Client, grpcConn *grpc.ClientConn,
	modelInputCallback models.GenerateModelInferRequest,
	modelOutputCallback models.GenerateModelInferOutputRequest,
	modelInferCallback nvidia_inferenceserver.DecoderFunc,
) (*BertModelService, error) {
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
	baseSrv := models.ModelService{
		MaxSeqLength:                    DefaultMaxSeqLength,
		TritonService:                   nvidia_inferenceserver.NewTritonClientForAll(httpAddr, httpClient, grpcConn),
		InferCallback:                   modelInferCallback,
		GenerateModelInferRequest:       modelInputCallback,
		GenerateModelInferOutputRequest: modelOutputCallback,
	}

	srv := &BertModelService{
		ModelService:  baseSrv,
		BertVocab:     voc,
		BertTokenizer: NewWordPieceTokenizer(voc),
	}
	return srv, nil
}
