package transformers

import (
	"slices"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"
	"github.com/sunhailin-Leo/triton-service-go/utils"
)

type W2NerModelService struct {
	*BertModelService
}

var dis2idx = []int32{
	0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
	9,
}

// like python list(range(start, len))
func generateRange(start, end int) []int {
	result := make([]int, end-start)

	for i := 0; i < end-start; i++ {
		result[i] = start + i
	}

	return result
}

func generateInitDistInputs(size int) [][]int32 {
	matrix := make([][]int32, size)
	for i := range matrix {
		matrix[i] = make([]int32, size)
	}

	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			matrix[i][j] = int32(i - j)
		}
	}

	return matrix
}

// getBertInputFeature Get Bert-W2NER Feature (before Make HTTP or GRPC Request).
func (w *W2NerModelService) getBertInputFeature(batchInferData [][]string) []*W2NERInputFeature {
	batchInputFeatures := make([]*W2NERInputFeature, len(batchInferData))
	batchInferTokens := make([][][]string, len(batchInferData))
	batchInferPieces := make([][]string, len(batchInferData))

	for i, inferData := range batchInferData {
		tokens := make([][]string, len(inferData))
		for j, token := range inferData {
			tokens[j] = w.getTokenizerResult(token)
		}
		batchInferTokens[i] = tokens
		batchInferPieces[i] = utils.Flatten2DSlice(tokens)
		batchInputFeatures[i] = &W2NERInputFeature{}
	}

	// Add CLS + SEP
	padPiecesLen := utils.GetMaxSubSliceLength(batchInferPieces) + 2
	for i, pieces := range batchInferPieces {
		inputIds := make([]int32, len(pieces))
		for j, token := range pieces {
			inputIds[j] = int32(w.BertVocab.GetID(token))
		}
		inputIds = slices.Insert(inputIds, 0, int32(w.BertVocab.GetID(DefaultCLS)))
		inputIds = slices.Insert(inputIds, len(inputIds), int32(w.BertVocab.GetID(DefaultSEP)))
		inputIds = utils.PadSlice(inputIds, padPiecesLen, 0)
		batchInputFeatures[i].TokenIDs = inputIds
	}

	padTokenLen := utils.GetMaxSubSliceLength(batchInferTokens)
	// pieces2word and gridMask2d
	for i, inferTokens := range batchInferTokens {
		gridMask2d := make([][]bool, padTokenLen)
		pieces2word := make([][]bool, padTokenLen)

		start := 0
		for j := 0; j < padTokenLen; j++ {
			gridMask2d[j] = make([]bool, padTokenLen)

			if j+1 >= len(inferTokens) {
				pieces2word[j] = make([]bool, len(batchInputFeatures[i].TokenIDs))
				continue
			}

			if len(inferTokens[j]) == 0 {
				continue
			}

			idx := generateRange(start, start+len(inferTokens[j]))
			pieces2word[j] = make([]bool, len(batchInputFeatures[i].TokenIDs))
			for k := 0; k < len(pieces2word[j]); k++ {
				if k >= idx[0]+1 && k < idx[len(idx)-1]+2 {
					pieces2word[j][k] = true
				}
			}
			start += len(inferTokens[j])
		}

		batchInputFeatures[i].GridMask2D = gridMask2d
		batchInputFeatures[i].Pieces2Word = pieces2word
	}

	// distInputs
	for i, inferTokens := range batchInferTokens {
		// len(inferTokens)
		distInputs := generateInitDistInputs(len(inferTokens))
		for j := 0; j < len(inferTokens); j++ {
			for k := 0; k < len(inferTokens); k++ {
				if distInputs[j][k] < 0 {
					distInputs[j][k] = dis2idx[-distInputs[j][k]] + 9
					if distInputs[j][k] == 0 {
						distInputs[j][k] = 19
					}
				} else {
					distInputs[j][k] = dis2idx[distInputs[j][k]]
					if distInputs[j][k] == 0 {
						distInputs[j][k] = 19
					}
				}
			}
			if len(distInputs[j]) < padTokenLen {
				distInputs[j] = utils.PadSlice(distInputs[j], padTokenLen, 0)
			}
		}
		if len(distInputs) < padTokenLen {
			distInputs = utils.PadSlice(distInputs, padTokenLen, make([]int32, padTokenLen))
		}

		batchInputFeatures[i].DistInputs = distInputs
	}

	return batchInputFeatures
}

// generateHTTPInputs get bert input feature for http request
// inferDataArr: model infer data slice
// inferInputs: triton inference server input tensor.
func (w *W2NerModelService) generateHTTPInputs(
	inferDataArr [][]string,
	inferInputs []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor,
) ([]HTTPBatchInput, []*W2NERInputFeature) {
	// Model Feature
	batchRequestInputs := make([]HTTPBatchInput, len(inferInputs))

	batchInputFeatures := w.getBertInputFeature(inferDataArr)
	inferDataObjs := make([][]any, len(batchInputFeatures))
	for i := range inferDataArr {
		inferDataObjs[i] = []any{
			batchInputFeatures[i].TokenIDs,
			batchInputFeatures[i].GridMask2D,
			batchInputFeatures[i].DistInputs,
			batchInputFeatures[i].Pieces2Word,
		}
	}
	inferDataObjs = utils.SliceTransposeFor2D(inferDataObjs)

	inferShapes := make([][]int64, len(inferInputs))
	for i := range inferInputs {
		switch i {
		case 0:
			inferShapes[i] = []int64{
				int64(len(inferDataArr)),
				int64(len(batchInputFeatures[0].TokenIDs)),
			}
		case 1, 2:
			inferShapes[i] = []int64{
				int64(len(inferDataArr)),
				int64(len(batchInputFeatures[0].GridMask2D)),
				int64(len(batchInputFeatures[0].GridMask2D[0])),
			}
		case 3:
			inferShapes[i] = []int64{
				int64(len(inferDataArr)),
				int64(len(batchInputFeatures[0].Pieces2Word)),
				int64(len(batchInputFeatures[0].Pieces2Word[0])),
			}
		}
	}

	for i := range inferInputs {
		batchRequestInputs[i] = HTTPBatchInput{
			Name:     inferInputs[i].Name,
			Shape:    inferShapes[i],
			DataType: inferInputs[i].Datatype,
			Data:     inferDataObjs[i],
		}
	}

	return batchRequestInputs, batchInputFeatures
}

// generateHTTPRequest HTTP Request Data Generate.
func (w *W2NerModelService) generateHTTPRequest(
	inferDataArr [][]string,
	inferInputs []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor,
	inferOutputs []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor,
) ([]byte, []*W2NERInputFeature, error) {
	// Generate batch request json body
	requestInputBody, modelInputObj := w.generateHTTPInputs(inferDataArr, inferInputs)
	jsonBody, jsonEncodeErr := w.TritonService.JsonMarshal(&HTTPRequestBody{
		Inputs:  requestInputBody,
		Outputs: w.generateHTTPOutputs(inferOutputs),
	})
	if jsonEncodeErr != nil {
		return nil, nil, jsonEncodeErr
	}
	return jsonBody, modelInputObj, nil
}

// ModelInfer API to call Triton Inference Server.
func (w *W2NerModelService) ModelInfer(
	inferData [][]string,
	modelName, modelVersion string,
	requestTimeout time.Duration,
	params ...interface{},
) ([]interface{}, error) {
	// Create request input/output tensors
	inferInputs := w.GenerateModelInferRequest()
	inferOutputs := w.GenerateModelInferOutputRequest(params...)

	// TODO GRPC

	httpRequestBody, httpInputData, err := w.generateHTTPRequest(inferData, inferInputs, inferOutputs)
	if err != nil {
		return nil, err
	}
	if httpRequestBody == nil {
		return nil, utils.ErrEmptyHTTPRequestBody
	}
	// HTTP Infer
	return w.TritonService.ModelHTTPInfer(
		httpRequestBody, modelName, modelVersion, requestTimeout,
		w.InferCallback, w, httpInputData, params,
	)
}
