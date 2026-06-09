package transformers

import (
	"context"
	"slices"

	"github.com/sunhailin-Leo/triton-service-go/v2/models"
	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
)

type W2NerModelService struct {
	*BertModelService
}

// dis2idx maps distance values to log-scale bucket indices.
// Pattern: 0→0, 1→1, 2..3→2, 4..7→3, 8..15→4, 16..31→5, 32..63→6, 64..127→7, 128..255→8, 256+→9
var dis2idx [1024]int32

func init() {
	// dis2idx[0] = 0 (zero-initialized)
	for i := 1; i < len(dis2idx); i++ {
		// bucket = floor(log2(i)) + 1, capped at 9
		bucket := int32(0)
		v := i
		for v > 0 {
			bucket++
			v >>= 1
		}
		// After loop: bucket = floor(log2(i)) + 1
		if bucket > 9 {
			bucket = 9
		}
		dis2idx[i] = bucket
	}
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

func generateAllTrueSlice(size int) []bool {
	matrix := make([]bool, size)
	for i := 0; i < size; i++ {
		matrix[i] = true
	}

	return matrix
}

///////////////////////////////////////// Bert Service Pre-Process Function /////////////////////////////////////////

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
		// The minus 2 is due to the retention of the CLS and SEP positions.
		tokens = utils.StringSliceTruncatePrecisely(tokens, w.MaxSeqLength-2)
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
			gridMask2d[j] = generateAllTrueSlice(padTokenLen)

			if j >= len(inferTokens) || len(inferTokens[j]) == 0 {
				pieces2word[j] = make([]bool, len(batchInputFeatures[i].TokenIDs))
				continue
			}

			idx := utils.GenerateRange[int](start, start+len(inferTokens[j]))
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

// Grpc2DSliceToLittleEndianByteSlice exports the private method for testing.
func (w *W2NerModelService) Grpc2DSliceToLittleEndianByteSlice(slice any, row, col int) []byte {
	return w.grpc2DSliceToLittleEndianByteSlice(slice, row, col)
}

// grpcSliceToLittleEndianByteSlice bool slice, 2-D int32 slice to byte slice with little endian.
func (w *W2NerModelService) grpc2DSliceToLittleEndianByteSlice(slice any, row, col int) []byte {
	switch s := slice.(type) {
	case [][]bool:
		returnByte := make([]byte, 0, row*col)
		for i := 0; i < row; i++ {
			returnByte = append(returnByte, w.grpcSliceToLittleEndianByteSlice(col, s[i], ModelBoolDataType)...)
		}
		return returnByte
	case [][]int32:
		returnByte := make([]byte, 0, row*col*4)
		for i := 0; i < row; i++ {
			returnByte = append(returnByte, w.grpcSliceToLittleEndianByteSlice(col, s[i], ModelInt32DataType)...)
		}
		return returnByte
	default:
		return nil
	}
}

// generateGRPCRequest GRPC Request Data Generate
func (w *W2NerModelService) generateGRPCRequest(
	inferDataArr [][]string,
	inferInputTensor []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor,
) ([][]byte, []*W2NERInputFeature) {
	var tokenIdsBytes, gridMask2DBytes, distInputsBytes, pieces2wordBytes []byte
	inputFeatures := w.getBertInputFeature(inferDataArr)
	for i := range inputFeatures {
		for j := range inferInputTensor {
			switch j {
			case 0:
				// TokenIDs []int32
				tokenIdsBytes = append(
					tokenIdsBytes,
					w.grpcSliceToLittleEndianByteSlice(
						len(inputFeatures[i].TokenIDs), inputFeatures[i].TokenIDs, inferInputTensor[0].Datatype)...,
				)
				inferInputTensor[j].Shape = []int64{int64(len(inferDataArr)), int64(len(inputFeatures[i].TokenIDs))}
			case 1:
				// GridMask2D [][]bool
				gridMask2DBytes = append(
					gridMask2DBytes,
					w.grpc2DSliceToLittleEndianByteSlice(inputFeatures[i].GridMask2D, len(inputFeatures[i].GridMask2D), len(inputFeatures[i].GridMask2D[0]))...,
				)
				inferInputTensor[j].Shape = []int64{int64(len(inferDataArr)), int64(len(inputFeatures[i].GridMask2D)), int64(len(inputFeatures[i].GridMask2D[0]))}
			case 2:
				// DistInputs [][]int32
				distInputsBytes = append(
					distInputsBytes,
					w.grpc2DSliceToLittleEndianByteSlice(inputFeatures[i].DistInputs, len(inputFeatures[i].DistInputs), len(inputFeatures[i].DistInputs[0]))...,
				)
				inferInputTensor[j].Shape = []int64{int64(len(inferDataArr)), int64(len(inputFeatures[i].DistInputs)), int64(len(inputFeatures[i].DistInputs[0]))}
			case 3:
				// Pieces2Word [][]bool
				pieces2wordBytes = append(
					pieces2wordBytes,
					w.grpc2DSliceToLittleEndianByteSlice(inputFeatures[i].Pieces2Word, len(inputFeatures[i].Pieces2Word), len(inputFeatures[i].Pieces2Word[0]))...,
				)
				inferInputTensor[j].Shape = []int64{int64(len(inferDataArr)), int64(len(inputFeatures[i].Pieces2Word)), int64(len(inputFeatures[i].Pieces2Word[0]))}
			}
		}
	}

	return [][]byte{tokenIdsBytes, gridMask2DBytes, distInputsBytes, pieces2wordBytes}, inputFeatures
}

///////////////////////////////////////// Bert Service Pre-Process Function /////////////////////////////////////////

//////////////////////////////////////////// Triton Service API Function ////////////////////////////////////////////

// ModelInfer API to call Triton Inference Server.
func (w *W2NerModelService) ModelInfer(
	ctx context.Context,
	inferData [][]string,
	modelName, modelVersion string,
	params ...any,
) ([]any, error) {
	// Create request input/output tensors
	inferInputs := w.GenerateModelInferRequest()
	inferOutputs := w.GenerateModelInferOutputRequest(params...)

	if w.IsGRPC {
		// GRPC Infer
		grpcRawInputs, grpcInputData := w.generateGRPCRequest(inferData, inferInputs)
		if grpcRawInputs == nil {
			return nil, utils.ErrEmptyGRPCRequestBody
		}
		return w.TritonService.ModelGRPCInfer(ctx, inferInputs, inferOutputs, grpcRawInputs, modelName, modelVersion,
			w.InferCallback, w, grpcInputData, params)
	}

	httpRequestBody, httpInputData, err := w.generateHTTPRequest(inferData, inferInputs, inferOutputs)
	if err != nil {
		return nil, err
	}
	if httpRequestBody == nil {
		return nil, utils.ErrEmptyHTTPRequestBody
	}
	// HTTP Infer
	return w.TritonService.ModelHTTPInfer(ctx, httpRequestBody, modelName, modelVersion, w.InferCallback,
		w, httpInputData, params)
}

//////////////////////////////////////////// Triton Service API Function ////////////////////////////////////////////

// W2NerOption configures a W2NerModelService.
type W2NerOption func(*W2NerModelService)

// WithW2NerMaxSeqLength sets the max sequence length.
func WithW2NerMaxSeqLength(maxSeqLen int) W2NerOption {
	return func(s *W2NerModelService) {
		s.MaxSeqLength = maxSeqLen
	}
}

// WithW2NerChineseTokenize enables Chinese tokenization.
func WithW2NerChineseTokenize(charMode bool) W2NerOption {
	return func(s *W2NerModelService) {
		s.IsChinese = true
		s.IsChineseCharMode = charMode
	}
}

// WithW2NerGRPCInfer enables gRPC for inference.
func WithW2NerGRPCInfer() W2NerOption {
	return func(s *W2NerModelService) {
		s.IsGRPC = true
	}
}

// WithW2NerTokenizerReturnPosInfo enables returning position info.
func WithW2NerTokenizerReturnPosInfo() W2NerOption {
	return func(s *W2NerModelService) {
		s.IsReturnPosArray = true
	}
}

// WithW2NerModelName sets the model name.
func WithW2NerModelName(prefix, name string) W2NerOption {
	return func(s *W2NerModelService) {
		s.ModelName = prefix + "-" + name
	}
}

// NewW2NERModelService creates a W2NerModelService with the given required parameters and optional configuration.
func NewW2NERModelService(
	bertVocabPath, httpAddr string,
	httpClient *fasthttp.Client, grpcConn *grpc.ClientConn,
	modelInputCallback models.GenerateModelInferRequest,
	modelOutputCallback models.GenerateModelInferOutputRequest,
	modelInferCallback nvidia_inferenceserver.DecoderFunc,
	opts ...W2NerOption,
) (*W2NerModelService, error) {
	// 1、Init Bert Service(Because W2NER is based on Bert)
	baseSrv, baseSrvErr := NewBertModelService(bertVocabPath, httpAddr, httpClient, grpcConn,
		modelInputCallback, modelOutputCallback, modelInferCallback)
	if baseSrvErr != nil {
		return nil, baseSrvErr
	}

	srv := &W2NerModelService{BertModelService: baseSrv}
	// 2、Apply options
	for _, opt := range opts {
		opt(srv)
	}
	return srv, nil
}
