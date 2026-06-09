package nvidia_inferenceserver_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// --- JSON Marshal/Unmarshal Benchmarks ---

func BenchmarkTritonClientService_JsonMarshal(b *testing.B) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	data := map[string]any{
		"inputs": []map[string]any{
			{"name": "input_ids", "shape": []int{1, 48}, "datatype": "INT32", "data": make([]int32, 48)},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = srv.JsonMarshal(data)
	}
}

func BenchmarkTritonClientService_JsonUnmarshal(b *testing.B) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	body, _ := json.Marshal(map[string]any{
		"model_name":    "bert",
		"model_version": "1",
		"outputs":       []map[string]any{{"name": "output"}},
	})
	var result map[string]any
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = srv.JsonUnmarshal(body, &result)
	}
}

// --- Constructor Benchmarks ---

func BenchmarkNewTritonClientWithOnlyHTTP(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	}
}

func BenchmarkNewTritonClientWithOnlyGRPC(b *testing.B) {
	conn, _ := grpc.NewClient("127.0.0.1:9000", grpc.WithTransportCredentials(insecure.NewCredentials()))
	defer func() { _ = conn.Close() }()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(conn)
	}
}

func BenchmarkNewTritonClientForAll(b *testing.B) {
	conn, _ := grpc.NewClient("127.0.0.1:9000", grpc.WithTransportCredentials(insecure.NewCredentials()))
	defer func() { _ = conn.Close() }()
	client := &fasthttp.Client{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = nvidia_inferenceserver.NewTritonClientForAll("127.0.0.1:9001", client, conn)
	}
}

// --- ensureCtx Benchmark ---

func BenchmarkTritonClientService_EnsureCtx_Background(b *testing.B) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := srv.EnsureCtx(context.Background())
		cancel()
		_ = ctx
	}
}

func BenchmarkTritonClientService_EnsureCtx_WithDeadline(b *testing.B) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		deadlineCtx, deadlineCancel := context.WithTimeout(context.Background(), 5*time.Second)
		ctx, cancel := srv.EnsureCtx(deadlineCtx)
		_ = ctx
		cancel()
		deadlineCancel()
	}
}

// --- pathEscape Benchmark ---

func BenchmarkPathEscape_Simple(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = nvidia_inferenceserver.PathEscape("mymodel")
	}
}

func BenchmarkPathEscape_SpecialChars(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = nvidia_inferenceserver.PathEscape("my/model name%test")
	}
}

// --- Error Handler Benchmarks ---

func BenchmarkHTTPErrorHandler_Nil(b *testing.B) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = srv.HTTPErrorHandler(200, nil)
	}
}

func BenchmarkGRPCErrorHandler_Nil(b *testing.B) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = srv.GRPCErrorHandler(nil)
	}
}
