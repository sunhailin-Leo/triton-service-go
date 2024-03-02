package test

import (
	"log"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/models/transformers"
	"github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
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
func testGenerateModelInferRequest() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor {
	return []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor{
		{
			Name:     tBertModelSegmentIdsKey,
			Datatype: tBertModelSegmentIdsDataType,
		},
		{
			Name:     tBertModelInputIdsKey,
			Datatype: tBertModelInputIdsDataType,
		},
		{
			Name:     tBertModelInputMaskKey,
			Datatype: tBertModelInputMaskDataType,
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
	bertService, initErr := transformers.NewBertModelService(
		vocabPath, httpAddr, defaultHTTPClient, defaultGRPCClient,
		testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback)
	if initErr != nil {
		panic(initErr)
	}
	bertService.SetChineseTokenize(false).SetMaxSeqLength(maxSeqLen)
	vocabSize := bertService.BertVocab.Size()
	if bertService.BertVocab.Size() != 21128 {
		t.Errorf("Expected '%d', but got '%d'", 21128, vocabSize)
	}
	if bertService.ModelService.MaxSeqLength != maxSeqLen {
		t.Errorf("Expected '%v', but got '%v'", maxSeqLen, bertService.ModelService.MaxSeqLength)
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
	bertService, initErr := transformers.NewBertModelService(
		vocabPath, httpAddr, defaultHTTPClient, defaultGRPCClient,
		testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback)
	if initErr != nil {
		panic(initErr)
	}
	bertService.SetMaxSeqLength(maxSeqLen)
	vocabSize := bertService.BertVocab.Size()
	if bertService.BertVocab.Size() != 119547 {
		t.Errorf("Expected '%d', but got '%d'", 119547, vocabSize)
	}
	if bertService.ModelService.MaxSeqLength != maxSeqLen {
		t.Errorf("Expected '%v', but got '%v'", maxSeqLen, bertService.ModelService.MaxSeqLength)
	}
}
