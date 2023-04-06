package test

import (
	"log"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/models/bert"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"
)

const (
	tBertModelSegmentIdsKey                       string = "segment_ids"
	tBertModelSegmentIdsDataType                  string = "INT32"
	tBertModelInputIdsKey                         string = "input_ids"
	tBertModelInputIdsDataType                    string = "INT32"
	tBertModelInputMaskKey                        string = "input_mask"
	tBertModelInputMaskDataType                   string = "INT32"
	tBertModelOutputProbabilitiesKey              string = "probability"
	tBertModelRespBodyOutputBinaryDataKey         string = "binary_data"
	tBertModelRespBodyOutputClassificationDataKey string = "classification"
)

// testGenerateModelInferRequest Triton Input.
func testGenerateModelInferRequest(
	batchSize, maxSeqLength int,
) []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor {
	return []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor{
		{
			Name:     tBertModelSegmentIdsKey,
			Datatype: tBertModelSegmentIdsDataType,
			Shape:    []int64{int64(batchSize), int64(maxSeqLength)},
		},
		{
			Name:     tBertModelInputIdsKey,
			Datatype: tBertModelInputIdsDataType,
			Shape:    []int64{int64(batchSize), int64(maxSeqLength)},
		},
		{
			Name:     tBertModelInputMaskKey,
			Datatype: tBertModelInputMaskDataType,
			Shape:    []int64{int64(batchSize), int64(maxSeqLength)},
		},
	}
}

// testGenerateModelInferOutputRequest Triton Output.
func testGenerateModelInferOutputRequest(
	params ...interface{},
) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor {
	for _, param := range params {
		log.Println("Param: ", param)
	}
	return []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor{
		{
			Name: tBertModelOutputProbabilitiesKey,
			Parameters: map[string]*nvidia_inferenceserver.InferParameter{
				tBertModelRespBodyOutputBinaryDataKey: {
					ParameterChoice: &nvidia_inferenceserver.InferParameter_BoolParam{BoolParam: false},
				},
				tBertModelRespBodyOutputClassificationDataKey: {
					ParameterChoice: &nvidia_inferenceserver.InferParameter_Int64Param{Int64Param: 1},
				},
			},
		},
	}
}

// testModerInferCallback infer call back (process model infer data).
func testModerInferCallback(inferResponse interface{}, params ...interface{}) ([]interface{}, error) {
	log.Println(inferResponse)
	log.Println(params...)
	return nil, nil
}

func TestBertServiceForBertChinese(t *testing.T) {
	vocabPath := "bert-chinese-vocab.txt"
	maxSeqLen := 48
	httpAddr := "127.0.0.1:9001"
	grpcAddr := "127.0.0.1:9000"
	defaultHTTPClient := &fasthttp.Client{}
	defaultGRPCClient, grpcErr := grpc.Dial(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if grpcErr != nil {
		panic(grpcErr)
	}
	bertService, initErr := bert.NewModelService(
		vocabPath, httpAddr, defaultHTTPClient, defaultGRPCClient,
		testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback)
	if initErr != nil {
		panic(initErr)
	}
	bertService = bertService.SetChineseTokenize().SetMaxSeqLength(maxSeqLen)
	vocabSize := bertService.BertVocab.Size()
	if bertService.BertVocab.Size() != 21128 {
		t.Errorf("Expected '%d', but got '%d'", vocabSize, 21128)
	}
}

func TestBertServiceForBertMultilingual(t *testing.T) {
	vocabPath := "bert-multilingual-vocab.txt"
	maxSeqLen := 64
	httpAddr := "127.0.0.1:9001"
	grpcAddr := "127.0.0.1:9000"
	defaultHTTPClient := &fasthttp.Client{}
	defaultGRPCClient, grpcErr := grpc.Dial(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if grpcErr != nil {
		panic(grpcErr)
	}
	bertService, initErr := bert.NewModelService(
		vocabPath, httpAddr, defaultHTTPClient, defaultGRPCClient,
		testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback)
	if initErr != nil {
		panic(initErr)
	}
	bertService = bertService.SetMaxSeqLength(maxSeqLen)
	vocabSize := bertService.BertVocab.Size()
	if bertService.BertVocab.Size() != 119547 {
		t.Errorf("Expected '%d', but got '%d'", vocabSize, 119547)
	}
}
