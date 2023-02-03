package bert

import "github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"

// GenerateModelInferRequest model input callback
type GenerateModelInferRequest func(batchSize, maxSeqLength int) []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor

// GenerateModelInferOutputRequest model output callback
type GenerateModelInferOutputRequest func() []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor

// InputFeature Bert InputFeature
type InputFeature struct {
	Text     string   // origin text
	Tokens   []string // token. like CLS/SEP after tokenizer
	TokenIDs []int32  // input_ids
	Mask     []int32  // input_mast
	TypeIDs  []int32  // segment_ids
}

// InputObjects bert input objects for position record
type InputObjects struct {
	Input  string
	Tokens []string
	//PosArray []string
}

// HTTPBatchInput Model HTTP Batch Request Input Struct.(Support batch 1)
type HTTPBatchInput struct {
	Name     string    `json:"name"`
	Shape    []int64   `json:"shape"`
	DataType string    `json:"datatype"`
	Data     [][]int32 `json:"data"`
}

// InferOutputParameter triton inference server infer parameters
type InferOutputParameter struct {
	BinaryData     bool  `json:"binary_data"`
	Classification int64 `json:"classification"`
}

// HTTPOutput Model HTTP Request Output Struct
type HTTPOutput struct {
	Name       string               `json:"name"`
	Parameters InferOutputParameter `json:"parameters"`
}

// HTTPRequestBody Model HTTP Request Body
type HTTPRequestBody struct {
	Inputs  []HTTPBatchInput `json:"inputs"`
	Outputs []HTTPOutput     `json:"outputs"`
}
