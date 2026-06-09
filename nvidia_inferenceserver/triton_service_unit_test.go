package nvidia_inferenceserver_test

import (
	"context"
	"net"
	"net/http"
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

// --- Error handler tests ---

func TestTritonClientService_ErrorHandler_Methods(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	t.Run("HTTPErrorHandler_NilErr_Non200", func(t *testing.T) {
		err := srv.HTTPErrorHandler(500, nil)
		if err == nil {
			t.Error("expected error for non-200 status with nil httpErr")
		}
	})

	t.Run("HTTPErrorHandler_NilErr_200", func(t *testing.T) {
		err := srv.HTTPErrorHandler(200, nil)
		if err != nil {
			t.Errorf("expected nil for 200 status with nil httpErr, got %v", err)
		}
	})

	t.Run("HTTPErrorHandler_WithErr", func(t *testing.T) {
		err := srv.HTTPErrorHandler(500, context.DeadlineExceeded)
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("GRPCErrorHandler", func(t *testing.T) {
		err := srv.GRPCErrorHandler(nil)
		if err != nil {
			t.Errorf("expected nil for nil grpcErr, got %v", err)
		}
	})

	t.Run("GRPCErrorHandler_WithErr", func(t *testing.T) {
		err := srv.GRPCErrorHandler(context.DeadlineExceeded)
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("DecodeFuncErrorHandler_HTTP", func(t *testing.T) {
		err := srv.DecodeFuncErrorHandler(context.DeadlineExceeded, false)
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("DecodeFuncErrorHandler_GRPC", func(t *testing.T) {
		err := srv.DecodeFuncErrorHandler(context.DeadlineExceeded, true)
		if err == nil {
			t.Error("expected error")
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
