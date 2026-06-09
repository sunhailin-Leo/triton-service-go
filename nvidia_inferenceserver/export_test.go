package nvidia_inferenceserver

import "context"

// Export internal methods for testing.

// EnsureCtx exports ensureCtx for testing.
func (t *TritonClientService) EnsureCtx(ctx context.Context) (context.Context, context.CancelFunc) {
	return t.ensureCtx(ctx)
}

// HTTPErrorHandler exports httpErrorHandler for testing.
func (t *TritonClientService) HTTPErrorHandler(statusCode int, httpErr error) error {
	return t.httpErrorHandler(statusCode, httpErr)
}

// GRPCErrorHandler exports grpcErrorHandler for testing.
func (t *TritonClientService) GRPCErrorHandler(grpcErr error) error {
	return t.grpcErrorHandler(grpcErr)
}

// DecodeFuncErrorHandler exports decodeFuncErrorHandler for testing.
func (t *TritonClientService) DecodeFuncErrorHandler(err error, isGRPC bool) error {
	return t.decodeFuncErrorHandler(err, isGRPC)
}

// PathEscape exports pathEscape for testing.
func PathEscape(s string) string {
	return pathEscape(s)
}
