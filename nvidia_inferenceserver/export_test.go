package nvidia_inferenceserver

import "context"

// Export internal methods for testing.

// EnsureCtx exports ensureCtx for testing.
func (t *TritonClientService) EnsureCtx(ctx context.Context) (context.Context, context.CancelFunc) {
	return t.ensureCtx(ctx)
}

// HTTPError exports httpError for testing.
func HTTPError(op string, statusCode int, err error) error {
	return httpError(op, statusCode, err)
}

// GRPCError exports grpcError for testing.
func GRPCError(op string, err error) error {
	return grpcError(op, err)
}

// DecodeError exports decodeError for testing.
func DecodeError(op string, isGRPC bool, err error) error {
	return decodeError(op, isGRPC, err)
}

// PathEscape exports pathEscape for testing.
func PathEscape(s string) string {
	return pathEscape(s)
}

// EnsureScheme exports ensureScheme for testing.
func EnsureScheme(rawURL string) string {
	return ensureScheme(rawURL)
}

// NewTritonError exports newTritonError for testing structured errors.
func NewTritonError(protocol Protocol, op string, statusCode int, err error) *TritonError {
	return &TritonError{
		Protocol:   protocol,
		Op:         op,
		StatusCode: statusCode,
		Err:        err,
	}
}
