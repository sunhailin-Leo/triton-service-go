package test

import (
	"encoding/json"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// TestNewTritonClientWithOnlyHTTP_NilClient tests creating HTTP client with nil fasthttp client (should use defaults)
func TestNewTritonClientWithOnlyHTTP_NilClient(t *testing.T) {
	client := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	if client == nil {
		t.Fatal("expected non-nil client when passing nil http client")
	}
}

// TestNewTritonClientWithOnlyHTTP_CustomClient tests creating HTTP client with custom fasthttp client
func TestNewTritonClientWithOnlyHTTP_CustomClient(t *testing.T) {
	httpClient := &fasthttp.Client{MaxConnsPerHost: 100}
	client := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", httpClient)
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

// TestNewTritonClientWithOnlyGRPC_NilConn tests creating GRPC client with nil connection
func TestNewTritonClientWithOnlyGRPC_NilConn(t *testing.T) {
	client := nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(nil)
	if client != nil {
		t.Fatal("expected nil client when passing nil grpc connection")
	}
}

// TestNewTritonClientWithOnlyGRPC_ValidConn tests creating GRPC client with valid connection
func TestNewTritonClientWithOnlyGRPC_ValidConn(t *testing.T) {
	conn, err := grpc.NewClient("127.0.0.1:9000", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to create grpc client: %v", err)
	}
	defer func() { _ = conn.Close() }()

	client := nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(conn)
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

// TestNewTritonClientForAll_NilGRPC tests creating client with nil grpc connection (should not panic)
func TestNewTritonClientForAll_NilGRPC(t *testing.T) {
	httpClient := &fasthttp.Client{}
	client := nvidia_inferenceserver.NewTritonClientForAll("127.0.0.1:9001", httpClient, nil)
	if client == nil {
		t.Fatal("expected non-nil client even with nil grpc connection")
	}
}

// TestNewTritonClientForAll_NilHTTP tests creating client with nil http client (should use defaults)
func TestNewTritonClientForAll_NilHTTP(t *testing.T) {
	conn, err := grpc.NewClient("127.0.0.1:9000", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to create grpc client: %v", err)
	}
	defer func() { _ = conn.Close() }()

	client := nvidia_inferenceserver.NewTritonClientForAll("127.0.0.1:9001", nil, conn)
	if client == nil {
		t.Fatal("expected non-nil client even with nil http client")
	}
}

// TestNewTritonClientForAll_BothNil tests creating client with both nil
func TestNewTritonClientForAll_BothNil(t *testing.T) {
	client := nvidia_inferenceserver.NewTritonClientForAll("127.0.0.1:9001", nil, nil)
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

// TestTritonClientService_JsonMarshalUnmarshal tests default JSON encoder/decoder
func TestTritonClientService_JsonMarshalUnmarshal(t *testing.T) {
	client := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	data := map[string]string{"key": "value"}
	encoded, err := client.JsonMarshal(data)
	if err != nil {
		t.Fatalf("JsonMarshal failed: %v", err)
	}

	var decoded map[string]string
	if err := client.JsonUnmarshal(encoded, &decoded); err != nil {
		t.Fatalf("JsonUnmarshal failed: %v", err)
	}

	if decoded["key"] != "value" {
		t.Errorf("expected key=value, got key=%s", decoded["key"])
	}
}

// TestTritonClientService_SetCustomJsonEncoder tests setting custom JSON encoder
func TestTritonClientService_SetCustomJsonEncoder(t *testing.T) {
	client := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	called := false
	customEncoder := func(v interface{}) ([]byte, error) {
		called = true
		return json.Marshal(v)
	}

	client.SetJSONEncoder(customEncoder)
	_, err := client.JsonMarshal(map[string]string{"test": "data"})
	if err != nil {
		t.Fatalf("custom encoder failed: %v", err)
	}
	if !called {
		t.Error("custom encoder was not called")
	}
}

// TestTritonClientService_SetAPIRequestTimeout tests setting API timeout
func TestTritonClientService_SetAPIRequestTimeout(t *testing.T) {
	client := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	// Should not panic
	client.SetAPIRequestTimeout(10 * 1e9) // 10 seconds
}

// TestTritonClientService_SetSecondaryServerURL tests setting secondary URL
func TestTritonClientService_SetSecondaryServerURL(t *testing.T) {
	client := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	// Should not panic
	client.SetSecondaryServerURL("127.0.0.1:9002")
}

// TestTritonClientService_ShutdownTritonConnection tests shutdown
func TestTritonClientService_ShutdownTritonConnection(t *testing.T) {
	client := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)
	err := client.ShutdownTritonConnection()
	if err != nil {
		t.Fatalf("shutdown failed: %v", err)
	}
}
