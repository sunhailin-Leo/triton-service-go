package nvidia_inferenceserver_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
	"github.com/valyala/fasthttp"
)

// startTestHTTPServer starts a local fasthttp server that responds with the given status and body.
func startTestHTTPServer(t *testing.T, handler func(ctx *fasthttp.RequestCtx)) string {
	t.Helper()
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	go fasthttp.Serve(ln, handler) //nolint:errcheck
	t.Cleanup(func() { _ = ln.Close() })
	return ln.Addr().String()
}

// errorHandler responds with the given HTTP status code and error body.
func errorHandler(statusCode int, errMsg string) func(ctx *fasthttp.RequestCtx) {
	return func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(statusCode)
		ctx.SetBody([]byte(errMsg))
	}
}

// --- HTTP Server Endpoint Tests ---

func TestCheckServerAlive_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/health/live" {
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{"live":true}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	alive, err := srv.CheckServerAlive(context.Background())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !alive {
		t.Error("expected alive=true")
	}
}

func TestCheckServerAlive_HTTP_Error(t *testing.T) {
	addr := startTestHTTPServer(t, errorHandler(fasthttp.StatusInternalServerError, "internal error"))
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	alive, err := srv.CheckServerAlive(context.Background())
	if err == nil {
		t.Fatal("expected error for status 500")
	}
	if alive {
		t.Error("expected alive=false on error")
	}
}

func TestCheckServerReady_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/health/ready" {
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{"ready":true}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	ready, err := srv.CheckServerReady(context.Background())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !ready {
		t.Error("expected ready=true")
	}
}

func TestCheckModelReady_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/models/mymodel/versions/1/ready" {
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{"ready":true}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	ready, err := srv.CheckModelReady(context.Background(), "mymodel", "1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !ready {
		t.Error("expected ready=true")
	}
}

func TestServerMetadata_HTTP_Success(t *testing.T) {
	expectedResp := map[string]any{"name": "triton", "version": "2.0"}
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2" {
			data, _ := json.Marshal(expectedResp)
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody(data)
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ServerMetadata(context.Background())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestModelMetadataRequest_HTTP_Success(t *testing.T) {
	expectedResp := map[string]any{"name": "mymodel", "versions": []string{"1"}}
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/models/mymodel/versions/1" {
			data, _ := json.Marshal(expectedResp)
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody(data)
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ModelMetadataRequest(context.Background(), "mymodel", "1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestModelConfiguration_HTTP_Success(t *testing.T) {
	expectedResp := map[string]any{"name": "mymodel", "platform": "onnx"}
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/models/mymodel/versions/1/config" {
			data, _ := json.Marshal(expectedResp)
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody(data)
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ModelConfiguration(context.Background(), "mymodel", "1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestModelInferStats_HTTP_Success(t *testing.T) {
	expectedResp := map[string]any{"name": "mymodel"}
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/models/mymodel/versions/1/stats" {
			data, _ := json.Marshal(expectedResp)
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody(data)
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ModelInferStats(context.Background(), "mymodel", "1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestModelIndex_HTTP_Success(t *testing.T) {
	expectedModels := []map[string]any{{"name": "mymodel", "state": "READY"}}
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/repository/index" {
			data, _ := json.Marshal(expectedModels)
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody(data)
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ModelIndex(context.Background(), "", true)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestModelLoadWithHTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/repository/models/mymodel/load" {
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ModelLoadWithHTTP(context.Background(), "mymodel", nil)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestModelUnloadWithHTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/repository/models/mymodel/unload" {
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ModelUnloadWithHTTP(context.Background(), "mymodel", nil)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestShareCUDAMemoryStatus_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/cudasharememory/region/region1/status" {
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ShareCUDAMemoryStatus(context.Background(), "region1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestShareSystemMemoryStatus_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/systemsharememory/region/region1/status" {
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ShareSystemMemoryStatus(context.Background(), "region1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestShareCUDAMemoryRegister_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/cudasharememory/region/region1/register" {
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ShareCUDAMemoryRegister(context.Background(), "region1", []byte("handle"), 0, 1024)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestShareCUDAMemoryUnRegister_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/cudasharememory/region/region1/unregister" {
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ShareCUDAMemoryUnRegister(context.Background(), "region1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestShareSystemMemoryRegister_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/systemsharememory/region/region1/register" {
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ShareSystemMemoryRegister(context.Background(), "region1", "key1", 1024, 0)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestShareSystemMemoryUnRegister_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/systemsharememory/region/region1/unregister" {
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.ShareSystemMemoryUnRegister(context.Background(), "region1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestGetModelTracingSetting_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/models/mymodel/trace/setting" && string(ctx.Method()) == fasthttp.MethodGet {
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.GetModelTracingSetting(context.Background(), "mymodel")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestSetModelTracingSetting_HTTP_Success(t *testing.T) {
	addr := startTestHTTPServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/v2/models/mymodel/trace/setting" && string(ctx.Method()) == fasthttp.MethodPost {
			ctx.SetContentType("application/json")
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetBody([]byte(`{}`))
		} else {
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	})
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	resp, err := srv.SetModelTracingSetting(context.Background(), "mymodel", nil)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

// --- Error Path Tests ---

func TestCheckServerAlive_HTTP_ServerError(t *testing.T) {
	addr := startTestHTTPServer(t, errorHandler(fasthttp.StatusServiceUnavailable, "unavailable"))
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	alive, err := srv.CheckServerAlive(context.Background())
	if err == nil {
		t.Fatal("expected error for 503")
	}
	if alive {
		t.Error("expected alive=false on error")
	}
}

func TestModelMetadataRequest_HTTP_ServerError(t *testing.T) {
	addr := startTestHTTPServer(t, errorHandler(fasthttp.StatusNotFound, "not found"))
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	_, err := srv.ModelMetadataRequest(context.Background(), "nonexistent", "1")
	if err == nil {
		t.Fatal("expected error for 404")
	}
}

func TestModelConfiguration_HTTP_ServerError(t *testing.T) {
	addr := startTestHTTPServer(t, errorHandler(fasthttp.StatusNotFound, "not found"))
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	_, err := srv.ModelConfiguration(context.Background(), "nonexistent", "1")
	if err == nil {
		t.Fatal("expected error for 404")
	}
}

func TestModelInferStats_HTTP_ServerError(t *testing.T) {
	addr := startTestHTTPServer(t, errorHandler(fasthttp.StatusNotFound, "not found"))
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	_, err := srv.ModelInferStats(context.Background(), "nonexistent", "1")
	if err == nil {
		t.Fatal("expected error for 404")
	}
}

func TestModelLoadWithHTTP_ServerError(t *testing.T) {
	addr := startTestHTTPServer(t, errorHandler(fasthttp.StatusInternalServerError, "error"))
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	_, err := srv.ModelLoadWithHTTP(context.Background(), "mymodel", nil)
	if err == nil {
		t.Fatal("expected error for 500")
	}
}

func TestModelUnloadWithHTTP_ServerError(t *testing.T) {
	addr := startTestHTTPServer(t, errorHandler(fasthttp.StatusInternalServerError, "error"))
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(addr, nil)
	_, err := srv.ModelUnloadWithHTTP(context.Background(), "mymodel", nil)
	if err == nil {
		t.Fatal("expected error for 500")
	}
}

// --- Context Tests ---

func TestModelHTTPInfer_CancelledContext(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:1", nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := srv.ModelHTTPInfer(ctx, []byte(`{}`), "model", "1",
		func(response any, params ...any) ([]any, error) { return nil, nil })
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

// --- Error Handler Unit Tests (via export_test.go) ---

func TestHTTPErrorHandler_WithValue(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	err := srv.HTTPErrorHandler(500, fmt.Errorf("internal error"))
	if err == nil {
		t.Fatal("expected non-nil error")
	}
	if err.Error() != "[HTTP]code: 500; error: internal error" {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}

func TestHTTPErrorHandler_NilErr_Non200Status(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	err := srv.HTTPErrorHandler(503, nil)
	if err == nil {
		t.Fatal("expected non-nil error for non-200 status with nil error")
	}
	if err.Error() != "[HTTP]unexpected status code: 503" {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}

func TestHTTPErrorHandler_NilErr_200Status(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	err := srv.HTTPErrorHandler(200, nil)
	if err != nil {
		t.Errorf("expected nil error for 200 status with nil error, got: %v", err)
	}
}

func TestGRPCErrorHandler_WithValue(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	err := srv.GRPCErrorHandler(fmt.Errorf("grpc error"))
	if err == nil {
		t.Fatal("expected non-nil error")
	}
	if err.Error() != "[GRPC]error: grpc error" {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}

func TestDecodeFuncErrorHandler_HTTP(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	err := srv.DecodeFuncErrorHandler(fmt.Errorf("decode error"), false)
	if err == nil {
		t.Fatal("expected non-nil error")
	}
	if err.Error() != "[HTTP]decodeFunc error: decode error" {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}

func TestDecodeFuncErrorHandler_GRPC(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	err := srv.DecodeFuncErrorHandler(fmt.Errorf("decode error"), true)
	if err == nil {
		t.Fatal("expected non-nil error")
	}
	if err.Error() != "[GRPC]decodeFunc error: decode error" {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}

func TestEnsureCtx_Nil(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	ctx, cancel := srv.EnsureCtx(context.TODO())
	defer cancel()
	if ctx == nil {
		t.Fatal("expected non-nil context")
	}
	if _, ok := ctx.Deadline(); !ok {
		t.Error("expected context with deadline")
	}
}

func TestEnsureCtx_WithDeadline(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	deadlineCtx, deadlineCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer deadlineCancel()
	ctx, cancel := srv.EnsureCtx(deadlineCtx)
	defer cancel()
	if _, ok := ctx.Deadline(); !ok {
		t.Error("expected context with deadline")
	}
}

func TestEnsureCtx_Background(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	ctx, cancel := srv.EnsureCtx(context.Background())
	defer cancel()
	if ctx == nil {
		t.Fatal("expected non-nil context")
	}
	if _, ok := ctx.Deadline(); !ok {
		t.Error("expected context with deadline")
	}
}

func TestPathEscape_Simple(t *testing.T) {
	result := nvidia_inferenceserver.PathEscape("mymodel")
	if result != "mymodel" {
		t.Errorf("expected 'mymodel', got %s", result)
	}
}

func TestPathEscape_SpecialChars(t *testing.T) {
	result := nvidia_inferenceserver.PathEscape("my/model")
	if result == "my/model" {
		t.Errorf("expected escaped path, got %s", result)
	}
}

// --- GRPC nil client tests ---

func TestModelGRPCInfer_NilGRPCClient(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	_, err := srv.ModelGRPCInfer(
		context.Background(),
		nil, nil, nil, "model", "1",
		func(response any, params ...any) ([]any, error) { return nil, nil },
	)
	if err == nil {
		t.Fatal("expected error for nil gRPC client")
	}
}

func TestModelGRPCInfer_PrivateModelGRPCInfer_NilClient(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	// The public ModelGRPCInfer calls modelGRPCInfer which returns error for nil client
	_, err := srv.ModelGRPCInfer(
		context.Background(),
		[]*nvidia_inferenceserver.ModelInferRequest_InferInputTensor{},
		[]*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor{},
		[][]byte{},
		"model", "1",
		func(response any, params ...any) ([]any, error) { return nil, nil },
	)
	if err == nil {
		t.Fatal("expected error for nil gRPC client")
	}
}

func TestModelLoadWithGRPC_NilGRPCClient(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	_, err := srv.ModelLoadWithGRPC(context.Background(), "repo", "model", nil)
	if err == nil {
		t.Fatal("expected error for nil gRPC client")
	}
}

func TestModelUnloadWithGRPC_NilGRPCClient(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	_, err := srv.ModelUnloadWithGRPC(context.Background(), "repo", "model", nil)
	if err == nil {
		t.Fatal("expected error for nil gRPC client")
	}
}

// --- Connection management tests ---

func TestShutdownTritonConnection_HTTPOnly(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	err := srv.ShutdownTritonConnection()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestGetServerURL_Primary(t *testing.T) {
	// Can't call getServerURL directly, but we can test via CheckServerAlive
	// which uses getServerURL and fails with connection refused
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:1", nil)
	// This will fail with connection refused, which exercises getServerURL
	_, _ = srv.CheckServerAlive(context.Background()) // just verify no panic
}

func TestGetServerURL_Secondary(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:1", nil)
	srv.SetSecondaryServerURL("127.0.0.1:2")
	// Just verify no panic; the actual HTTP request will fail
	_, _ = srv.CheckServerAlive(context.Background())
}

// --- JSON Custom Encoder/Decoder Tests with HTTP ---

func TestSetJSONEncoder_WithHTTP(t *testing.T) {
	called := false
	customEncoder := func(v any) ([]byte, error) {
		called = true
		return json.Marshal(v)
	}
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:1", nil)
	srv.SetJSONEncoder(customEncoder)
	_, err := srv.JsonMarshal(map[string]string{"key": "value"})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !called {
		t.Error("expected custom encoder to be called")
	}
}

func TestSetJSONDecoder_WithHTTP(t *testing.T) {
	called := false
	customDecoder := func(data []byte, v any) error {
		called = true
		return json.Unmarshal(data, v)
	}
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:1", nil)
	srv.SetJSONDecoder(customDecoder)
	var result map[string]string
	err := srv.JsonUnmarshal([]byte(`{"key":"value"}`), &result)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !called {
		t.Error("expected custom decoder to be called")
	}
}
