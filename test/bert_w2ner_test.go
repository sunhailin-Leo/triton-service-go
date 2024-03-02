package test

import (
	"log"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/models/transformers"
	"github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"
	"github.com/sunhailin-Leo/triton-service-go/utils"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func testGenerateW2NERModelInferRequest() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor {
	return []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor{
		{
			Name:     "input_ids",
			Datatype: utils.TritonINT32Type,
			// Shape: [batch_size, input_ids_shape[0]},
		},
		{
			Name:     "grid_mask2d",
			Datatype: utils.TritonBoolType,
			// Shape: [batch_size, grid_mask2d_shape[0], grid_mask2d_shape[1]]
		},
		{
			Name:     "dist_inputs",
			Datatype: utils.TritonINT32Type,
			// Shape: [batch_size, dist_inputs_shape[0], dist_inputs_shape[1]]
		},
		{
			Name:     "pieces2word",
			Datatype: utils.TritonBoolType,
			// Shape: [batch_size, pieces2word_shape[0], pieces2word_shape[1]]
		},
	}
}

func testGenerateW2NERModelInferOutputRequest(params ...interface{}) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor {
	for _, param := range params {
		log.Println("Param: ", param)
	}
	return []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor{
		{
			Name: "logic",
			// Parameters: map[string]*nvidia_inferenceserver.InferParameter{
			//	 tBertModelRespBodyOutputBinaryDataKey: {
			//		 ParameterChoice: &nvidia_inferenceserver.InferParameter_BoolParam{BoolParam: false},
			//	 },
			// },
		},
	}
}

// testModerInferCallback infer call back (process model infer data).
func testW2NERModerInferCallback(inferResponse interface{}, params ...interface{}) ([]interface{}, error) {
	log.Println("response: ", inferResponse)
	log.Println(params...)
	return nil, nil
}

func TestW2NERService(t *testing.T) {
	vocabPath := "bert-multilingual-vocab.txt"

	httpAddr := "127.0.0.1:9001"
	grpcAddr := "127.0.0.1:9000"

	defaultHTTPClient := &fasthttp.Client{}
	defaultGRPCClient, grpcErr := grpc.Dial(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if grpcErr != nil {
		panic(grpcErr)
	}

	bertService, initErr := transformers.NewBertModelService(
		vocabPath, httpAddr, defaultHTTPClient, defaultGRPCClient,
		testGenerateW2NERModelInferRequest,
		testGenerateW2NERModelInferOutputRequest,
		testW2NERModerInferCallback)
	if initErr != nil {
		panic(initErr)
	}

	w2nerService := &transformers.W2NerModelService{BertModelService: bertService}
	vocabSize := w2nerService.BertVocab.Size()
	if w2nerService.BertVocab.Size() != 119547 {
		t.Errorf("Expected '%d', but got '%d'", 119547, vocabSize)
	}
}
