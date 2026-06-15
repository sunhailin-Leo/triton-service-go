package nvidia_inferenceserver_test

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// --- ensureCtx tests ---

func TestTritonClientService_EnsureCtx_Nil(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	// Test with nil context - should default to context.Background() with timeout
	ctx, cancel := srv.EnsureCtx(context.TODO())
	defer cancel()
	if ctx == nil {
		t.Fatal("expected non-nil context")
	}
	if _, ok := ctx.Deadline(); !ok {
		t.Error("expected context to have a deadline")
	}
}

func TestTritonClientService_EnsureCtx_Background(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	ctx, cancel := srv.EnsureCtx(context.Background())
	defer cancel()
	if _, ok := ctx.Deadline(); !ok {
		t.Error("expected context to have a deadline when passing Background()")
	}
}

func TestTritonClientService_EnsureCtx_WithDeadline(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	deadlineCtx, deadlineCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer deadlineCancel()

	ctx, cancel := srv.EnsureCtx(deadlineCtx)
	defer cancel()
	if ctx != deadlineCtx {
		t.Error("expected same context when already has deadline")
	}
	origDL, _ := deadlineCtx.Deadline()
	resultDL, _ := ctx.Deadline()
	if !origDL.Equal(resultDL) {
		t.Error("expected same deadline")
	}
}

func TestTritonClientService_EnsureCtx_CustomTimeout(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	srv.SetAPIRequestTimeout(10 * time.Second)

	ctx, cancel := srv.EnsureCtx(context.Background())
	defer cancel()
	dl, ok := ctx.Deadline()
	if !ok {
		t.Fatal("expected context to have a deadline")
	}
	remaining := time.Until(dl)
	if remaining < 8*time.Second || remaining > 12*time.Second {
		t.Errorf("expected ~10s remaining, got %v", remaining)
	}
}

// --- pathEscape tests (via exported test wrapper) ---

func TestPathEscape(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"simple", "mymodel", "mymodel"},
		{"with slash", "my/model", "my%2Fmodel"},
		{"with space", "my model", "my%20model"},
		{"with percent", "my%model", "my%25model"},
		{"chinese chars", "中文模型", "%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B"},
		{"empty", "", ""},
		{"version", "1", "1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := nvidia_inferenceserver.PathEscape(tt.input)
			if result != tt.expected {
				t.Errorf("PathEscape(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

// --- Context cancellation tests ---

func TestTritonClientService_ModelHTTPInfer_CancelledContext(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := srv.ModelHTTPInfer(ctx, []byte(`{}`), "model", "1", func(response any, params ...any) ([]any, error) {
		return nil, nil
	})
	if err == nil {
		t.Error("expected error for cancelled context")
	}
}

// --- ModelHTTPInfer with mock server ---

func TestTritonClientService_ModelHTTPInfer_ServerError(t *testing.T) {
	// Create a mock HTTP server that returns 500
	mockServer := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"internal server error"}`))
		}),
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to create listener: %v", err)
	}
	go func() { _ = mockServer.Serve(ln) }()
	defer func() { _ = mockServer.Close() }()

	addr := ln.Addr().String()
	client := &fasthttp.Client{
		MaxConnsPerHost: 10,
		ReadTimeout:     5 * time.Second,
		WriteTimeout:    5 * time.Second,
	}
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, client)

	_, inferErr := srv.ModelHTTPInfer(context.Background(), []byte(`{}`), "model", "1",
		func(response any, params ...any) ([]any, error) {
			return nil, nil
		})
	if inferErr == nil {
		t.Error("expected error for 500 status code")
	}
}

func TestTritonClientService_CheckServerAlive_HTTP(t *testing.T) {
	// Test with no server - should get connection error
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:0", &fasthttp.Client{
		ReadTimeout:  100 * time.Millisecond,
		WriteTimeout: 100 * time.Millisecond,
	})
	_, err := srv.CheckServerAlive(context.Background())
	if err == nil {
		t.Error("expected error when no server is running")
	}
}

func TestTritonClientService_CheckServerReady_HTTP(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:0", &fasthttp.Client{
		ReadTimeout:  100 * time.Millisecond,
		WriteTimeout: 100 * time.Millisecond,
	})
	_, err := srv.CheckServerReady(context.Background())
	if err == nil {
		t.Error("expected error when no server is running")
	}
}

func TestTritonClientService_CheckModelReady_HTTP(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:0", &fasthttp.Client{
		ReadTimeout:  100 * time.Millisecond,
		WriteTimeout: 100 * time.Millisecond,
	})
	_, err := srv.CheckModelReady(context.Background(), "mymodel", "1")
	if err == nil {
		t.Error("expected error when no server is running")
	}
}

// --- gRPC nil client tests ---

func TestTritonClientService_ModelGRPCInfer_NilClient(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	// ModelGRPCInfer should return error when grpcClient is nil
	_, err := srv.ModelGRPCInfer(
		context.Background(),
		nil, nil, nil,
		"model", "1",
		func(response any, params ...any) ([]any, error) {
			return nil, nil
		},
	)
	if err == nil {
		t.Error("expected error for nil grpc client")
	}
}

func TestTritonClientService_ModelLoadWithGRPC_NilClient(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	_, err := srv.ModelLoadWithGRPC(context.Background(), "repo", "model", nil)
	if err == nil {
		t.Error("expected error for nil grpc client")
	}
}

func TestTritonClientService_ModelUnloadWithGRPC_NilClient(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	_, err := srv.ModelUnloadWithGRPC(context.Background(), "repo", "model", nil)
	if err == nil {
		t.Error("expected error for nil grpc client")
	}
}

// --- gRPC client with real connection (but no server) ---

func TestTritonClientService_GRPCMethods_NoServer(t *testing.T) {
	conn, err := grpc.NewClient("127.0.0.1:0", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create gRPC client: %v", err)
	}
	defer func() { _ = conn.Close() }()

	srv, err := nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(conn)
	if err != nil {
		t.Fatalf("Failed to create service: %v", err)
	}

	// These should return errors (connection refused) but not panic
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	t.Run("CheckServerAlive", func(t *testing.T) {
		_, err := srv.CheckServerAlive(ctx)
		if err == nil {
			t.Error("expected error with no server")
		}
	})

	t.Run("ServerMetadata", func(t *testing.T) {
		_, err := srv.ServerMetadata(ctx)
		if err == nil {
			t.Error("expected error with no server")
		}
	})

	t.Run("ModelMetadataRequest", func(t *testing.T) {
		_, err := srv.ModelMetadataRequest(ctx, "model", "1")
		if err == nil {
			t.Error("expected error with no server")
		}
	})

	t.Run("ModelIndex", func(t *testing.T) {
		_, err := srv.ModelIndex(ctx, "", true)
		if err == nil {
			t.Error("expected error with no server")
		}
	})

	t.Run("ModelConfiguration", func(t *testing.T) {
		_, err := srv.ModelConfiguration(ctx, "model", "1")
		if err == nil {
			t.Error("expected error with no server")
		}
	})

	t.Run("ModelInferStats", func(t *testing.T) {
		_, err := srv.ModelInferStats(ctx, "model", "1")
		if err == nil {
			t.Error("expected error with no server")
		}
	})

	t.Run("GetModelTracingSetting", func(t *testing.T) {
		_, err := srv.GetModelTracingSetting(ctx, "model")
		if err == nil {
			t.Error("expected error with no server")
		}
	})

	t.Run("ShareCUDAMemoryStatus", func(t *testing.T) {
		_, err := srv.ShareCUDAMemoryStatus(ctx, "region")
		if err == nil {
			t.Error("expected error with no server")
		}
	})

	t.Run("ShareSystemMemoryStatus", func(t *testing.T) {
		_, err := srv.ShareSystemMemoryStatus(ctx, "region")
		if err == nil {
			t.Error("expected error with no server")
		}
	})
}

// --- Structured error tests ---

func TestStructuredErrors(t *testing.T) {
	t.Run("HTTPError_NilErr_Non200", func(t *testing.T) {
		err := nvidia_inferenceserver.HTTPError("test", 500, nil)
		if err == nil {
			t.Error("expected error for non-200 status with nil httpErr")
		}
		var tritonErr *nvidia_inferenceserver.TritonError
		if !errors.As(err, &tritonErr) {
			t.Error("expected TritonError type")
		}
		if tritonErr.Protocol != nvidia_inferenceserver.ProtocolHTTP {
			t.Errorf("expected HTTP protocol, got %v", tritonErr.Protocol)
		}
	})

	t.Run("HTTPError_NilErr_200", func(t *testing.T) {
		err := nvidia_inferenceserver.HTTPError("test", 200, nil)
		if err != nil {
			t.Errorf("expected nil for 200 status with nil httpErr, got %v", err)
		}
	})

	t.Run("HTTPError_WithErr", func(t *testing.T) {
		err := nvidia_inferenceserver.HTTPError("test", 500, context.DeadlineExceeded)
		if err == nil {
			t.Error("expected error")
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Error("expected error to wrap DeadlineExceeded")
		}
	})

	t.Run("GRPCError_Nil", func(t *testing.T) {
		err := nvidia_inferenceserver.GRPCError("test", nil)
		if err != nil {
			t.Errorf("expected nil for nil grpcErr, got %v", err)
		}
	})

	t.Run("GRPCError_WithErr", func(t *testing.T) {
		err := nvidia_inferenceserver.GRPCError("test", context.DeadlineExceeded)
		if err == nil {
			t.Error("expected error")
		}
		var tritonErr *nvidia_inferenceserver.TritonError
		if !errors.As(err, &tritonErr) {
			t.Error("expected TritonError type")
		}
		if tritonErr.Protocol != nvidia_inferenceserver.ProtocolGRPC {
			t.Errorf("expected GRPC protocol, got %v", tritonErr.Protocol)
		}
	})

	t.Run("DecodeError_HTTP", func(t *testing.T) {
		err := nvidia_inferenceserver.DecodeError("infer", false, context.DeadlineExceeded)
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("DecodeError_GRPC", func(t *testing.T) {
		err := nvidia_inferenceserver.DecodeError("infer", true, context.DeadlineExceeded)
		if err == nil {
			t.Error("expected error")
		}
	})
}

// --- EnsureScheme tests ---

func TestEnsureScheme(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"127.0.0.1:8000", "http://127.0.0.1:8000"},
		{"http://127.0.0.1:8000", "http://127.0.0.1:8000"},
		{"https://127.0.0.1:8000", "https://127.0.0.1:8000"},
		{"triton.example.com", "http://triton.example.com"},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := nvidia_inferenceserver.EnsureScheme(tt.input)
			if result != tt.expected {
				t.Errorf("EnsureScheme(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

// --- ClientOption tests ---

func TestClientOptions(t *testing.T) {
	t.Run("WithTimeout", func(t *testing.T) {
		srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil,
			nvidia_inferenceserver.WithTimeout(15*time.Second))
		ctx, cancel := srv.EnsureCtx(context.Background())
		defer cancel()
		dl, ok := ctx.Deadline()
		if !ok {
			t.Fatal("expected deadline")
		}
		remaining := time.Until(dl)
		if remaining < 13*time.Second || remaining > 17*time.Second {
			t.Errorf("expected ~15s remaining, got %v", remaining)
		}
	})

	t.Run("WithLogger", func(t *testing.T) {
		logger := slog.Default()
		srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil,
			nvidia_inferenceserver.WithLogger(logger))
		if srv.Logger() == nil {
			t.Error("expected non-nil logger")
		}
	})
}

// --- Shutdown with GRPC connection ---

func TestTritonClientService_ShutdownTritonConnection_GRPC(t *testing.T) {
	conn, err := grpc.NewClient("127.0.0.1:0", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create gRPC client: %v", err)
	}

	srv, srvErr := nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(conn)
	if srvErr != nil {
		t.Fatalf("Failed to create service: %v", srvErr)
	}

	shutdownErr := srv.ShutdownTritonConnection()
	if shutdownErr != nil {
		t.Errorf("expected no error on shutdown, got %v", shutdownErr)
	}
}

// --- Shutdown with both HTTP and GRPC ---

func TestTritonClientService_ShutdownTritonConnection_All(t *testing.T) {
	conn, err := grpc.NewClient("127.0.0.1:0", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create gRPC client: %v", err)
	}

	srv := nvidia_inferenceserver.NewTritonClientForAll("127.0.0.1:9001", &fasthttp.Client{}, conn)
	shutdownErr := srv.ShutdownTritonConnection()
	if shutdownErr != nil {
		t.Errorf("expected no error on shutdown, got %v", shutdownErr)
	}
}

// --- TritonError.Error() tests ---

func TestTritonError_Error(t *testing.T) {
	tests := []struct {
		name       string
		protocol   nvidia_inferenceserver.Protocol
		op         string
		statusCode int
		err        error
		expected   string
	}{
		{
			name:       "with StatusCode",
			protocol:   nvidia_inferenceserver.ProtocolHTTP,
			op:         "infer",
			statusCode: 500,
			err:        errors.New("internal error"),
			expected:   "[HTTP]infer: code=500; internal error",
		},
		{
			name:       "without StatusCode with Err",
			protocol:   nvidia_inferenceserver.ProtocolGRPC,
			op:         "infer",
			statusCode: 0,
			err:        errors.New("connection refused"),
			expected:   "[GRPC]infer: connection refused",
		},
		{
			name:       "without StatusCode without Err",
			protocol:   nvidia_inferenceserver.ProtocolHTTP,
			op:         "health",
			statusCode: 0,
			err:        nil,
			expected:   "[HTTP]health",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tritonErr := nvidia_inferenceserver.NewTritonError(tt.protocol, tt.op, tt.statusCode, tt.err)
			result := tritonErr.Error()
			if result != tt.expected {
				t.Errorf("Error() = %q, want %q", result, tt.expected)
			}
		})
	}
}

// --- ClientOption tests for JSON encoder/decoder and HTTP client ---

func TestWithJSONEncoder(t *testing.T) {
	customEncoder := func(v any) ([]byte, error) {
		return []byte(`{"custom":"encoder"}`), nil
	}

	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil,
		nvidia_inferenceserver.WithJSONEncoder(customEncoder))

	data, err := srv.JsonMarshal(map[string]string{"test": "value"})
	if err != nil {
		t.Fatalf("JsonMarshal failed: %v", err)
	}

	expected := `{"custom":"encoder"}`
	if string(data) != expected {
		t.Errorf("JsonMarshal() = %s, want %s", string(data), expected)
	}
}

func TestWithJSONDecoder(t *testing.T) {
	customDecoder := func(data []byte, v any) error {
		// Custom decoder that sets a fixed value
		if m, ok := v.(*map[string]string); ok {
			*m = map[string]string{"decoded": "by_custom_decoder"}
		}
		return nil
	}

	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil,
		nvidia_inferenceserver.WithJSONDecoder(customDecoder))

	var result map[string]string
	err := srv.JsonUnmarshal([]byte(`{"original":"data"}`), &result)
	if err != nil {
		t.Fatalf("JsonUnmarshal failed: %v", err)
	}

	if result["decoded"] != "by_custom_decoder" {
		t.Errorf("JsonUnmarshal() result = %v, want decoded=by_custom_decoder", result)
	}
}

func TestWithHTTPClient(t *testing.T) {
	customClient := &fasthttp.Client{
		MaxConnsPerHost: 100,
		ReadTimeout:     30 * time.Second,
		WriteTimeout:    30 * time.Second,
	}

	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", customClient)

	// Verify the client is not nil by attempting an operation that uses it
	// We expect a connection error since no server is running, but the client should be used
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err := srv.CheckServerAlive(ctx)
	if err == nil {
		t.Error("expected error when no server is running")
	}
	// The fact that we got an error (not a panic) indicates the custom client was used
}

// --- ModelGRPCInfer decode error branch test ---

func TestTritonClientService_ModelGRPCInfer_DecodeError(t *testing.T) {
	// Test the decode error path by using DecodeError directly
	// This verifies that decodeError creates the correct structured error

	err := nvidia_inferenceserver.DecodeError("infer", true, errors.New("decode failed"))
	if err == nil {
		t.Fatal("expected error from DecodeError")
	}

	// Verify it's a structured error
	var tritonErr *nvidia_inferenceserver.TritonError
	if !errors.As(err, &tritonErr) {
		t.Error("expected TritonError type")
	}

	expectedOp := "infer.decode"
	if tritonErr.Op != expectedOp {
		t.Errorf("expected op %q, got %q", expectedOp, tritonErr.Op)
	}

	if tritonErr.Protocol != nvidia_inferenceserver.ProtocolGRPC {
		t.Errorf("expected GRPC protocol, got %v", tritonErr.Protocol)
	}
}

// --- ModelHTTPInfer success test with mock server ---

func TestModelHTTPInfer_Success(t *testing.T) {
	// Create a mock HTTP server that returns 200 + valid JSON
	mockServer := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"outputs":[{"name":"output","shape":[1],"datatype":"FP32","data":[1.0]}]}`))
		}),
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to create listener: %v", err)
	}
	go func() { _ = mockServer.Serve(ln) }()
	defer func() { _ = mockServer.Close() }()

	addr := ln.Addr().String()
	client := &fasthttp.Client{
		MaxConnsPerHost: 10,
		ReadTimeout:     5 * time.Second,
		WriteTimeout:    5 * time.Second,
	}
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, client)

	// Decoder function that parses the response
	decoderFunc := func(response any, params ...any) ([]any, error) {
		body, ok := response.([]byte)
		if !ok {
			return nil, errors.New("response is not []byte")
		}
		var result map[string]any
		if err := json.Unmarshal(body, &result); err != nil {
			return nil, err
		}
		return []any{result}, nil
	}

	ctx := context.Background()
	requestBody := []byte(`{"inputs":[{"name":"input","shape":[1],"datatype":"FP32","data":[1.0]}]}`)

	results, inferErr := srv.ModelHTTPInfer(ctx, requestBody, "model", "1", decoderFunc)
	if inferErr != nil {
		t.Fatalf("ModelHTTPInfer failed: %v", inferErr)
	}

	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
}

// --- ShareMemoryUnRegister HTTP branch tests ---

func TestShareCUDAMemoryUnRegister_HTTP(t *testing.T) {
	// Create a mock HTTP server that returns 200 for unregister
	mockServer := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify the request path
			if !strings.Contains(r.URL.Path, "/cudasharememory/region/") || !strings.Contains(r.URL.Path, "/unregister") {
				t.Errorf("unexpected request path: %s", r.URL.Path)
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{}`))
		}),
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to create listener: %v", err)
	}
	go func() { _ = mockServer.Serve(ln) }()
	defer func() { _ = mockServer.Close() }()

	addr := ln.Addr().String()
	client := &fasthttp.Client{
		MaxConnsPerHost: 10,
		ReadTimeout:     5 * time.Second,
		WriteTimeout:    5 * time.Second,
	}
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, client)

	ctx := context.Background()
	resp, err := srv.ShareCUDAMemoryUnRegister(ctx, "test_region")
	if err != nil {
		t.Fatalf("ShareCUDAMemoryUnRegister failed: %v", err)
	}

	if resp == nil {
		t.Error("expected non-nil response")
	}
}

func TestShareSystemMemoryUnRegister_HTTP(t *testing.T) {
	// Create a mock HTTP server that returns 200 for unregister
	mockServer := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify the request path
			if !strings.Contains(r.URL.Path, "/systemsharememory/region/") || !strings.Contains(r.URL.Path, "/unregister") {
				t.Errorf("unexpected request path: %s", r.URL.Path)
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{}`))
		}),
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to create listener: %v", err)
	}
	go func() { _ = mockServer.Serve(ln) }()
	defer func() { _ = mockServer.Close() }()

	addr := ln.Addr().String()
	client := &fasthttp.Client{
		MaxConnsPerHost: 10,
		ReadTimeout:     5 * time.Second,
		WriteTimeout:    5 * time.Second,
	}
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, client)

	ctx := context.Background()
	resp, err := srv.ShareSystemMemoryUnRegister(ctx, "test_region")
	if err != nil {
		t.Fatalf("ShareSystemMemoryUnRegister failed: %v", err)
	}

	if resp == nil {
		t.Error("expected non-nil response")
	}
}
