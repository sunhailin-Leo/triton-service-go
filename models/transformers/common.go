package transformers

// InputFeature Bert InputFeature.
type InputFeature struct {
	Text     string   // origin text
	Tokens   []string // token. like CLS/SEP after tokenizer
	TokenIDs []int32  // input_ids
	Mask     []int32  // input_mask
	TypeIDs  []int32  // segment_ids
}

// InputObjects bert input objects for position record.
type InputObjects struct {
	Input    string
	Tokens   []string
	PosArray []OffsetsType
}

type W2NERInputFeature struct {
	Text        string
	TokenIDs    []int32   // input_ids
	GridMask2D  [][]bool  // grid_mask2d
	DistInputs  [][]int32 // disk_inputs
	Pieces2Word [][]bool  // pieces2word
}

// HTTPBatchInput Model HTTP Batch Request Input Struct (Support batch 1).
type HTTPBatchInput struct {
	Name     string  `json:"name"`
	Shape    []int64 `json:"shape"`
	DataType string  `json:"datatype"`
	Data     any     `json:"data"`
}

// InferOutputParameter triton inference server infer parameters.
type InferOutputParameter struct {
	BinaryData     bool  `json:"binary_data,omitempty"`
	Classification int64 `json:"classification,omitempty"`
}

// HTTPOutput Model HTTP Request Output Struct.
type HTTPOutput struct {
	Name       string               `json:"name"`
	Parameters InferOutputParameter `json:"parameters,omitempty"`
}

// HTTPRequestBody Model HTTP Request Body.
type HTTPRequestBody struct {
	Inputs  []HTTPBatchInput `json:"inputs"`
	Outputs []HTTPOutput     `json:"outputs"`
}
