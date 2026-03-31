package transformers

// GrpcSliceToLittleEndianByteSlice exports the private method for testing only.
// This file uses package transformers (not transformers_test) so it can access unexported methods,
// and the _test.go suffix ensures it is only compiled during testing.
func (m *BertModelService) GrpcSliceToLittleEndianByteSlice(maxLen int, input any, inputType string) []byte {
	return m.grpcSliceToLittleEndianByteSlice(maxLen, input, inputType)
}
