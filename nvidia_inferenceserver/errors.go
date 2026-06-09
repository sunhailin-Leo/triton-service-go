package nvidia_inferenceserver

import (
	"fmt"
)

// Protocol represents the communication protocol used.
type Protocol string

const (
	ProtocolHTTP Protocol = "HTTP"
	ProtocolGRPC Protocol = "GRPC"
)

// TritonError is a structured error type for Triton client operations.
type TritonError struct {
	// Protocol is the communication protocol (HTTP or GRPC).
	Protocol Protocol
	// Op is the operation that failed (e.g., "infer", "health", "load").
	Op string
	// StatusCode is the HTTP status code (0 for gRPC errors).
	StatusCode int
	// Err is the underlying error.
	Err error
}

// Error implements the error interface.
func (e *TritonError) Error() string {
	if e.StatusCode > 0 {
		return fmt.Sprintf("[%s]%s: code=%d; %v", e.Protocol, e.Op, e.StatusCode, e.Err)
	}
	if e.Err != nil {
		return fmt.Sprintf("[%s]%s: %v", e.Protocol, e.Op, e.Err)
	}
	return fmt.Sprintf("[%s]%s", e.Protocol, e.Op)
}

// Unwrap returns the underlying error for errors.Is/As support.
func (e *TritonError) Unwrap() error {
	return e.Err
}

// newHTTPError creates a TritonError for HTTP operations.
func newHTTPError(op string, statusCode int, err error) *TritonError {
	return &TritonError{Protocol: ProtocolHTTP, Op: op, StatusCode: statusCode, Err: err}
}

// newGRPCError creates a TritonError for gRPC operations.
func newGRPCError(op string, err error) *TritonError {
	return &TritonError{Protocol: ProtocolGRPC, Op: op, Err: err}
}
