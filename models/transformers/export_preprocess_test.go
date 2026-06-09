package transformers

import "github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"

// ExportBertGetBertInputFeature exports the private method getBertInputFeature for testing.
func (m *BertModelService) ExportBertGetBertInputFeature(inferData string) (*InputFeature, *InputObjects) {
	return m.getBertInputFeature(inferData)
}

// ExportBertGenerateHTTPRequest exports the private method generateHTTPRequest for testing.
func (m *BertModelService) ExportBertGenerateHTTPRequest(
	inferDataArr []string,
	inferInputs []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor,
	inferOutputs []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor,
) ([]byte, []*InputObjects, error) {
	return m.generateHTTPRequest(inferDataArr, inferInputs, inferOutputs)
}

// ExportBertGenerateGRPCRequest exports the private method generateGRPCRequest for testing.
func (m *BertModelService) ExportBertGenerateGRPCRequest(
	inferDataArr []string,
	inferInputTensor []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor,
) ([][]byte, []*InputObjects) {
	return m.generateGRPCRequest(inferDataArr, inferInputTensor)
}

// ExportW2NerGenerateInitDistInputs exports the package-level function generateInitDistInputs for testing.
func ExportW2NerGenerateInitDistInputs(size int) [][]int32 {
	return generateInitDistInputs(size)
}

// ExportW2NerGenerateAllTrueSlice exports the package-level function generateAllTrueSlice for testing.
func ExportW2NerGenerateAllTrueSlice(size int) []bool {
	return generateAllTrueSlice(size)
}
