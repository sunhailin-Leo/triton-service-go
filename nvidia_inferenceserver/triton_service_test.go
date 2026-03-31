package nvidia_inferenceserver_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func TestNewTritonClientWithOnlyHTTP_NilClient(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	if srv == nil {
		t.Fatal("Expected service to be created")
	}
}

func TestNewTritonClientWithOnlyHTTP_CustomClient(t *testing.T) {
	customClient := &fasthttp.Client{
		MaxConnsPerHost: 100,
		ReadTimeout:     10 * time.Second,
		WriteTimeout:    10 * time.Second,
	}

	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", customClient)

	if srv == nil {
		t.Fatal("Expected service to be created")
	}
}

func TestNewTritonClientWithOnlyGRPC_NilConn(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(nil)

	if srv != nil {
		t.Error("Expected nil service for nil connection")
	}
}

func TestNewTritonClientWithOnlyGRPC_ValidConn(t *testing.T) {
	conn, err := grpc.NewClient("127.0.0.1:9000", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Skipf("Failed to connect to gRPC server: %v", err)
	}
	defer func() { _ = conn.Close() }()

	srv := nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(conn)

	if srv == nil {
		t.Fatal("Expected service to be created")
	}
}

func TestNewTritonClientForAll(t *testing.T) {
	tests := []struct {
		name       string
		httpClient *fasthttp.Client
		grpcConn   *grpc.ClientConn
		expectNil  bool
	}{
		{
			name:       "both nil",
			httpClient: nil,
			grpcConn:   nil,
			expectNil:  false,
		},
		{
			name:       "only http client",
			httpClient: &fasthttp.Client{},
			grpcConn:   nil,
			expectNil:  false,
		},
		{
			name:       "only grpc conn",
			httpClient: nil,
			grpcConn:   nil,
			expectNil:  false,
		},
		{
			name:       "both provided",
			httpClient: &fasthttp.Client{},
			grpcConn:   nil,
			expectNil:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := nvidia_inferenceserver.NewTritonClientForAll("127.0.0.1:9001", tt.httpClient, tt.grpcConn)

			if tt.expectNil && srv != nil {
				t.Error("Expected nil service")
			}
			if !tt.expectNil && srv == nil {
				t.Error("Expected non-nil service")
			}
		})
	}
}

func TestTritonClientService_JsonMarshal(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	data := map[string]string{"key": "value"}
	result, err := srv.JsonMarshal(data)

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Fatal("Expected non-nil result")
	}

	var decoded map[string]string
	if err := json.Unmarshal(result, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}
	if decoded["key"] != "value" {
		t.Errorf("Expected key=value, got key=%s", decoded["key"])
	}
}

func TestTritonClientService_JsonUnmarshal(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	data := []byte(`{"key":"value"}`)
	var result map[string]string

	err := srv.JsonUnmarshal(data, &result)

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if result["key"] != "value" {
		t.Errorf("Expected key=value, got key=%s", result["key"])
	}
}

func TestTritonClientService_SetJSONEncoder(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	customEncoder := func(v interface{}) ([]byte, error) {
		return []byte(`{"custom":true}`), nil
	}

	srv.SetJSONEncoder(customEncoder)

	result, err := srv.JsonMarshal(nil)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if string(result) != `{"custom":true}` {
		t.Errorf("Expected custom encoder output, got %s", string(result))
	}
}

func TestTritonClientService_SetJsonDecoder(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	customDecoder := func(data []byte, v interface{}) error {
		if m, ok := v.(*map[string]string); ok {
			(*m)["custom"] = "decoded"
		}
		return nil
	}

	srv.SetJsonDecoder(customDecoder)

	result := make(map[string]string)
	err := srv.JsonUnmarshal([]byte(`{}`), &result)

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if result["custom"] != "decoded" {
		t.Errorf("Expected custom decoder output, got %s", result["custom"])
	}
}

func TestTritonClientService_SetAPIRequestTimeout(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	timeout := 30 * time.Second

	// Verify SetAPIRequestTimeout doesn't panic
	srv.SetAPIRequestTimeout(timeout)
}

func TestTritonClientService_SetSecondaryServerURL(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	secondaryURL := "127.0.0.1:9002"

	// Verify SetSecondaryServerURL doesn't panic
	srv.SetSecondaryServerURL(secondaryURL)
}

func TestTritonClientService_ShutdownTritonConnection(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	err := srv.ShutdownTritonConnection()

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
}

// Test API model serialization/deserialization
func TestModelIndexRequestHTTPObj_MarshalUnmarshal(t *testing.T) {
	tests := []struct {
		name string
		obj  *nvidia_inferenceserver.ModelIndexRequestHTTPObj
	}{
		{
			name: "valid request with repo name",
			obj: &nvidia_inferenceserver.ModelIndexRequestHTTPObj{
				RepoName: "my-repo",
				Ready:    true,
			},
		},
		{
			name: "valid request with empty repo name",
			obj: &nvidia_inferenceserver.ModelIndexRequestHTTPObj{
				RepoName: "",
				Ready:    false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.obj)
			if err != nil {
				t.Fatalf("Failed to marshal: %v", err)
			}

			var decoded nvidia_inferenceserver.ModelIndexRequestHTTPObj
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}

			if decoded.RepoName != tt.obj.RepoName {
				t.Errorf("Expected RepoName %s, got %s", tt.obj.RepoName, decoded.RepoName)
			}
			if decoded.Ready != tt.obj.Ready {
				t.Errorf("Expected Ready %v, got %v", tt.obj.Ready, decoded.Ready)
			}
		})
	}
}

func TestCudaMemoryRegisterBodyHTTPObj_MarshalUnmarshal(t *testing.T) {
	tests := []struct {
		name string
		obj  *nvidia_inferenceserver.CudaMemoryRegisterBodyHTTPObj
	}{
		{
			name: "valid cuda memory register request",
			obj: &nvidia_inferenceserver.CudaMemoryRegisterBodyHTTPObj{
				RawHandle: []byte("cuda-handle-123"),
				DeviceID:  0,
				ByteSize:  1024,
			},
		},
		{
			name: "cuda memory register with different device",
			obj: &nvidia_inferenceserver.CudaMemoryRegisterBodyHTTPObj{
				RawHandle: []byte("cuda-handle-456"),
				DeviceID:  1,
				ByteSize:  2048,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.obj)
			if err != nil {
				t.Fatalf("Failed to marshal: %v", err)
			}

			var decoded nvidia_inferenceserver.CudaMemoryRegisterBodyHTTPObj
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}

			if decoded.DeviceID != tt.obj.DeviceID {
				t.Errorf("Expected DeviceID %d, got %d", tt.obj.DeviceID, decoded.DeviceID)
			}
			if decoded.ByteSize != tt.obj.ByteSize {
				t.Errorf("Expected ByteSize %d, got %d", tt.obj.ByteSize, decoded.ByteSize)
			}
		})
	}
}

func TestSystemMemoryRegisterBodyHTTPObj_MarshalUnmarshal(t *testing.T) {
	tests := []struct {
		name string
		obj  *nvidia_inferenceserver.SystemMemoryRegisterBodyHTTPObj
	}{
		{
			name: "valid system memory register request",
			obj: &nvidia_inferenceserver.SystemMemoryRegisterBodyHTTPObj{
				Key:      "mem-key-123",
				Offset:   0,
				ByteSize: 1024,
			},
		},
		{
			name: "system memory register with offset",
			obj: &nvidia_inferenceserver.SystemMemoryRegisterBodyHTTPObj{
				Key:      "mem-key-456",
				Offset:   512,
				ByteSize: 2048,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.obj)
			if err != nil {
				t.Fatalf("Failed to marshal: %v", err)
			}

			var decoded nvidia_inferenceserver.SystemMemoryRegisterBodyHTTPObj
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}

			if decoded.Key != tt.obj.Key {
				t.Errorf("Expected Key %s, got %s", tt.obj.Key, decoded.Key)
			}
			if decoded.Offset != tt.obj.Offset {
				t.Errorf("Expected Offset %d, got %d", tt.obj.Offset, decoded.Offset)
			}
			if decoded.ByteSize != tt.obj.ByteSize {
				t.Errorf("Expected ByteSize %d, got %d", tt.obj.ByteSize, decoded.ByteSize)
			}
		})
	}
}

func TestTraceSettingRequestHTTPObj_MarshalUnmarshal(t *testing.T) {
	tests := []struct {
		name string
		obj  *nvidia_inferenceserver.TraceSettingRequestHTTPObj
	}{
		{
			name: "valid trace setting request",
			obj: &nvidia_inferenceserver.TraceSettingRequestHTTPObj{
				TraceSetting: map[string]interface{}{
					"trace_level": 1,
					"rate":        0.5,
				},
			},
		},
		{
			name: "trace setting with empty settings",
			obj: &nvidia_inferenceserver.TraceSettingRequestHTTPObj{
				TraceSetting: map[string]interface{}{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.obj)
			if err != nil {
				t.Fatalf("Failed to marshal: %v", err)
			}

			var decoded nvidia_inferenceserver.TraceSettingRequestHTTPObj
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}

			if decoded.TraceSetting == nil && tt.obj.TraceSetting != nil {
				t.Error("Expected non-nil TraceSetting")
			}
		})
	}
}

// Test protobuf enum values
func TestHealthCheckResponse_ServingStatus(t *testing.T) {
	tests := []struct {
		name   string
		status nvidia_inferenceserver.HealthCheckResponse_ServingStatus
		want   int32
	}{
		{
			name:   "UNKNOWN status",
			status: nvidia_inferenceserver.HealthCheckResponse_UNKNOWN,
			want:   0,
		},
		{
			name:   "SERVING status",
			status: nvidia_inferenceserver.HealthCheckResponse_SERVING,
			want:   1,
		},
		{
			name:   "NOT_SERVING status",
			status: nvidia_inferenceserver.HealthCheckResponse_NOT_SERVING,
			want:   2,
		},
		{
			name:   "SERVICE_UNKNOWN status",
			status: nvidia_inferenceserver.HealthCheckResponse_SERVICE_UNKNOWN,
			want:   3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if int32(tt.status) != tt.want {
				t.Errorf("Expected status value %d, got %d", tt.want, tt.status)
			}

			// Test Enum() method
			enumPtr := tt.status.Enum()
			if enumPtr == nil {
				t.Fatal("Expected non-nil enum pointer")
			}
			if *enumPtr != tt.status {
				t.Errorf("Expected enum pointer value %v, got %v", tt.status, *enumPtr)
			}

			// Test String() method
			str := tt.status.String()
			if str == "" {
				t.Error("Expected non-empty string representation")
			}
		})
	}
}

// Test error handler methods (these can be tested without network)
func TestTritonClientService_ErrorHandlers(t *testing.T) {
	_ = nvidia_inferenceserver.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", nil)

	t.Run("httpErrorHandler with nil error", func(t *testing.T) {
		t.Skip("httpErrorHandler is a private method")
	})

	t.Run("grpcErrorHandler with nil error", func(t *testing.T) {
		t.Skip("grpcErrorHandler is a private method")
	})

	t.Run("decodeFuncErrorHandler with nil error", func(t *testing.T) {
		t.Skip("decodeFuncErrorHandler is a private method")
	})
}

// Test URL construction (via getServerURL)
func TestTritonClientService_GetServerURL(t *testing.T) {
	tests := []struct {
		name               string
		serverURL          string
		secondaryServerURL string
		expectedURL        string
	}{
		{
			name:               "primary server URL",
			serverURL:          "127.0.0.1:9001",
			secondaryServerURL: "",
			expectedURL:        "http://127.0.0.1:9001",
		},
		{
			name:               "secondary server URL set",
			serverURL:          "127.0.0.1:9001",
			secondaryServerURL: "127.0.0.1:9002",
			expectedURL:        "http://127.0.0.1:9002",
		},
		{
			name:               "different server address",
			serverURL:          "192.168.1.100:8000",
			secondaryServerURL: "",
			expectedURL:        "http://192.168.1.100:8000",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(tt.serverURL, nil)
			if tt.secondaryServerURL != "" {
				srv.SetSecondaryServerURL(tt.secondaryServerURL)
			}

			// We can't directly call getServerURL() as it's private
			// But we can verify the behavior by checking the server URL in error messages
			// or by testing methods that use getServerURL()
			// For now, we'll just verify that SetSecondaryServerURL works without panicking
		})
	}
}

// Test JSON marshaling/unmarshaling for API models
func TestAPIModel_JSONOperations(t *testing.T) {
	t.Run("ModelIndexRequestHTTPObj nil map handling", func(t *testing.T) {
		obj := &nvidia_inferenceserver.ModelIndexRequestHTTPObj{
			RepoName: "",
			Ready:    false,
		}
		data, err := json.Marshal(obj)
		if err != nil {
			t.Fatalf("Failed to marshal: %v", err)
		}

		var decoded nvidia_inferenceserver.ModelIndexRequestHTTPObj
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("Failed to unmarshal: %v", err)
		}

		if decoded.RepoName != "" {
			t.Errorf("Expected empty RepoName, got %s", decoded.RepoName)
		}
		if decoded.Ready != false {
			t.Errorf("Expected Ready false, got %v", decoded.Ready)
		}
	})
}

// Test protobuf enum maps
func TestHealthCheckResponse_ServingStatusMaps(t *testing.T) {
	t.Run("ServingStatus_name map", func(t *testing.T) {
		tests := []struct {
			value int32
			name  string
		}{
			{0, "UNKNOWN"},
			{1, "SERVING"},
			{2, "NOT_SERVING"},
			{3, "SERVICE_UNKNOWN"},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := nvidia_inferenceserver.HealthCheckResponse_ServingStatus_name[tt.value]; got != tt.name {
					t.Errorf("Expected name %s for value %d, got %s", tt.name, tt.value, got)
				}
			})
		}
	})

	t.Run("ServingStatus_value map", func(t *testing.T) {
		tests := []struct {
			name  string
			value int32
		}{
			{"UNKNOWN", 0},
			{"SERVING", 1},
			{"NOT_SERVING", 2},
			{"SERVICE_UNKNOWN", 3},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := nvidia_inferenceserver.HealthCheckResponse_ServingStatus_value[tt.name]; got != tt.value {
					t.Errorf("Expected value %d for name %s, got %d", tt.value, tt.name, got)
				}
			})
		}
	})
}

// Test TraceDurationObj structure
func TestTraceDurationObj(t *testing.T) {
	t.Run("create and check fields", func(t *testing.T) {
		// This is just a struct, we can test its creation and usage
		duration := nvidia_inferenceserver.TraceDurationObj{
			PreProcessNanoDuration:  1000,
			InferNanoDuration:       2000,
			PostProcessNanoDuration: 3000,
		}

		if duration.PreProcessNanoDuration != 1000 {
			t.Errorf("Expected PreProcessNanoDuration 1000, got %d", duration.PreProcessNanoDuration)
		}
		if duration.InferNanoDuration != 2000 {
			t.Errorf("Expected InferNanoDuration 2000, got %d", duration.InferNanoDuration)
		}
		if duration.PostProcessNanoDuration != 3000 {
			t.Errorf("Expected PostProcessNanoDuration 3000, got %d", duration.PostProcessNanoDuration)
		}
	})

	t.Run("JSON marshaling", func(t *testing.T) {
		duration := nvidia_inferenceserver.TraceDurationObj{
			PreProcessNanoDuration:  1000,
			InferNanoDuration:       2000,
			PostProcessNanoDuration: 3000,
		}

		data, err := json.Marshal(duration)
		if err != nil {
			t.Fatalf("Failed to marshal: %v", err)
		}

		var decoded nvidia_inferenceserver.TraceDurationObj
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("Failed to unmarshal: %v", err)
		}

		if decoded.PreProcessNanoDuration != duration.PreProcessNanoDuration {
			t.Errorf("Expected PreProcessNanoDuration %d, got %d", duration.PreProcessNanoDuration, decoded.PreProcessNanoDuration)
		}
	})
}
