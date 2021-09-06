package nvidia_inferenceserver

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/tidwall/gjson"
	"github.com/valyala/fasthttp"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

const (
	HTTPPrefix      string = "http"
	HttpPostMethod  string = "POST"
	JsonContentType string = "application/json"
)

// TritonGRPCService Service interface
type TritonGRPCService interface {
	// CheckServerAlive Check triton inference server is alive.
	CheckServerAlive(timeout time.Duration, isGRPC bool) (bool, error)
	// CheckServerReady Check triton inference server is ready.
	CheckServerReady(timeout time.Duration, isGRPC bool) (bool, error)
	// CheckModelReady Check triton inference server`s model is ready.
	CheckModelReady(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (bool, error)
	// ServerMetadata Get triton inference server metadata.
	ServerMetadata(timeout time.Duration, isGRPC bool) (*ServerMetadataResponse, error)
	// ModelGRPCInfer Call triton inference server infer with GRPC
	ModelGRPCInfer(inferInputs []*ModelInferRequest_InferInputTensor, inferOutputs []*ModelInferRequest_InferRequestedOutputTensor, rawInputs [][]byte, modelName, modelVersion string, timeout time.Duration, decoderFunc DecoderFunc) (interface{}, error)
	// ModelHTTPInfer all triton inference server infer with HTTP
	ModelHTTPInfer(requestBody []byte, modelName, modelVersion string, timeout time.Duration, decoderFunc DecoderFunc) (interface{}, error)
	// ModelMetadataRequest Get triton inference server`s model metadata.
	ModelMetadataRequest(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (*ModelMetadataResponse, error)
	// ModelIndex Get triton inference server model index.
	ModelIndex(isReady bool, timeout time.Duration, isGRPC bool) (*RepositoryIndexResponse, error)
	// ModelConfiguration Get triton inference server model configuration.
	ModelConfiguration(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (interface{}, error)
	// ModelInferStats Get triton inference server model infer stats.
	ModelInferStats(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (*ModelStatisticsResponse, error)
	// ModelLoad Load model
	ModelLoad(modelName string, timeout time.Duration, isGRPC bool) (*RepositoryModelLoadResponse, error)
	// ModelUnload Unload Model
	ModelUnload(modelName string, timeout time.Duration, isGRPC bool) (*RepositoryModelUnloadResponse, error)
	// ShareMemoryStatus Show share memory / share cuda memory status.
	ShareMemoryStatus(isCUDA bool, regionName string, timeout time.Duration, isGRPC bool) (interface{}, error)
	// ShareMemoryRegister Register share memory / share cuda memory). TODO Implement
	// https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_shared_memory.md
	ShareMemoryRegister(isCUDA bool, regionName string, timeout time.Duration, isGRPC bool) (interface{}, error)
	// ShareMemoryUnRegister Unregister share memory / share cuda memory). TODO Implement
	// https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_shared_memory.md
	ShareMemoryUnRegister(isCUDA bool, regionName string, timeout time.Duration, isGRPC bool) (interface{}, error)

	// ConnectToTritonWithGRPC : Connect to triton with GRPC
	ConnectToTritonWithGRPC() error
	// DisconnectToTritonWithGRPC : Close GRPC Connections
	DisconnectToTritonWithGRPC() error
	// ConnectToTritonWithHTTP : Create HTTP Client Pool
	ConnectToTritonWithHTTP()
	// DisconnectToTritonWithHTTP : Close HTTP Client Pool
	DisconnectToTritonWithHTTP()
}

// DecoderFunc Infer Callback Function
type DecoderFunc func(interface{}) (interface{}, error)

// TritonClientService ServiceClient
type TritonClientService struct {
	ServerURL string

	grpcConn   *grpc.ClientConn
	client     GRPCInferenceServiceClient
	httpClient *fasthttp.Client
}

// modelHTTPInferWithFiber Call Triton Inference Server with fiber HTTP（core function）
func (t *TritonClientService) modelHTTPInferWithFiber(modelName, modelVersion string, requestBody []byte, timeout time.Duration) (interface{}, error) {
	_, retBody, retErrors := fiber.Post(HTTPPrefix + "://" + t.ServerURL + "/v2/models/" + modelName + "/versions/" + modelVersion + "/infer").Timeout(timeout).Body(requestBody).Bytes()
	if len(retErrors) > 0 {
		return nil, retErrors[0]
	}
	return retBody, nil
}

// ModelHTTPInferWithFiber Call Triton with fiber HTTP
func (t *TritonClientService) ModelHTTPInferWithFiber(requestBody []byte, modelName, modelVersion string, timeout time.Duration, decoderFunc DecoderFunc) (interface{}, error) {
	// get infer response
	modelInferResponse, inferErr := t.modelHTTPInferWithFiber(modelName, modelVersion, requestBody, timeout)
	if inferErr != nil {
		return nil, fmt.Errorf("inferErr: " + inferErr.Error())
	}
	// decode Result
	response, decodeErr := decoderFunc(modelInferResponse)
	if decodeErr != nil {
		return nil, fmt.Errorf("decodeErr: " + decodeErr.Error())
	}
	return response, nil
}

// modelHTTPInfer Call Triton Inference Server with HTTP（core function）
func (t *TritonClientService) modelHTTPInfer(modelName, modelVersion string, requestBody []byte, timeout time.Duration) (interface{}, error) {
	// requestObj
	requestObj := fasthttp.AcquireRequest()
	requestObj.SetRequestURI(HTTPPrefix + "://" + t.ServerURL + "/v2/models/" + modelName + "/versions/" + modelVersion + "/infer")
	requestObj.Header.SetMethod(HttpPostMethod)
	requestObj.Header.SetContentType(JsonContentType)
	requestObj.SetBody(requestBody)
	// responseObj
	responseObj := fasthttp.AcquireResponse()
	httpErr := t.httpClient.DoTimeout(requestObj, responseObj, timeout)
	// for gc
	requestBody = nil
	if httpErr != nil {
		return nil, httpErr
	}
	// return response body
	return responseObj.Body(), nil
}

// ModelHTTPInfer Call Triton with HTTP
func (t *TritonClientService) ModelHTTPInfer(requestBody []byte, modelName, modelVersion string, timeout time.Duration, decoderFunc DecoderFunc) (interface{}, error) {
	// get infer response
	modelInferResponse, inferErr := t.modelHTTPInfer(modelName, modelVersion, requestBody, timeout)
	if inferErr != nil {
		return nil, fmt.Errorf("inferErr: " + inferErr.Error())
	}
	// decode Result
	response, decodeErr := decoderFunc(modelInferResponse)
	if decodeErr != nil {
		return nil, fmt.Errorf("decodeErr: " + decodeErr.Error())
	}
	return response, nil
}

// modelGRPCInfer Call Triton with GRPC（core function）
func (t *TritonClientService) modelGRPCInfer(inferRequest *ModelInferRequest, timeout time.Duration) (*ModelInferResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	// Get infer response
	modelInferResponse, inferErr := t.client.ModelInfer(ctx, inferRequest)
	if inferErr != nil {
		return nil, fmt.Errorf("inferErr: " + inferErr.Error())
	}
	return modelInferResponse, nil
}

// ModelGRPCInfer Call Triton with GRPC
func (t *TritonClientService) ModelGRPCInfer(inferInputs []*ModelInferRequest_InferInputTensor, inferOutputs []*ModelInferRequest_InferRequestedOutputTensor, rawInputs [][]byte, modelName, modelVersion string, timeout time.Duration, decoderFunc DecoderFunc) (interface{}, error) {
	// Create infer request for specific model/version
	modelInferRequest := ModelInferRequest{
		ModelName:        modelName,
		ModelVersion:     modelVersion,
		Inputs:           inferInputs,
		Outputs:          inferOutputs,
		RawInputContents: rawInputs,
	}
	// Get infer response
	modelInferResponse, inferErr := t.modelGRPCInfer(&modelInferRequest, timeout)
	if inferErr != nil {
		return nil, inferErr
	}
	// decode Result
	response, decodeErr := decoderFunc(modelInferResponse)
	if decodeErr != nil {
		return nil, decodeErr
	}
	return response, nil
}

// CheckServerAlive check server is alive
func (t *TritonClientService) CheckServerAlive(timeout time.Duration, isGRPC bool) (bool, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		serverLiveRequest := ServerLiveRequest{}
		serverLiveResponse, err := t.client.ServerLive(ctx, &serverLiveRequest)
		if err != nil {
			return false, err
		}
		return serverLiveResponse.Live, nil
	} else {
		requestObj := fasthttp.AcquireRequest()
		requestObj.SetRequestURI(fmt.Sprintf("%s://%s/v2/health/live", HTTPPrefix, t.ServerURL))
		responseObj := fasthttp.AcquireResponse()
		if err := t.httpClient.DoTimeout(requestObj, responseObj, timeout); err != nil {
			return false, err
		}
		defer fasthttp.ReleaseRequest(requestObj)
		defer fasthttp.ReleaseResponse(responseObj)
		if responseObj.StatusCode() != 200 {
			return false, fmt.Errorf("response code: %d", responseObj.StatusCode())
		}
		return true, nil
	}
}

// CheckServerReady check server is ready
func (t *TritonClientService) CheckServerReady(timeout time.Duration, isGRPC bool) (bool, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		serverReadyRequest := ServerReadyRequest{}
		serverReadyResponse, err := t.client.ServerReady(ctx, &serverReadyRequest)
		if err != nil {
			return false, err
		}
		return serverReadyResponse.Ready, nil
	} else {
		requestObj := fasthttp.AcquireRequest()
		requestObj.SetRequestURI(fmt.Sprintf("%s://%s/v2/health/ready", HTTPPrefix, t.ServerURL))
		responseObj := fasthttp.AcquireResponse()
		if err := t.httpClient.DoTimeout(requestObj, responseObj, timeout); err != nil {
			return false, err
		}
		defer fasthttp.ReleaseRequest(requestObj)
		defer fasthttp.ReleaseResponse(responseObj)
		if responseObj.StatusCode() != 200 {
			return false, fmt.Errorf("response code: %d", responseObj.StatusCode())
		}
		return true, nil
	}
}

// CheckModelReady check model is ready
func (t *TritonClientService) CheckModelReady(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (bool, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		modelReadyRequest := ModelReadyRequest{Name: modelName, Version: modelVersion}
		modelReadyResponse, err := t.client.ModelReady(ctx, &modelReadyRequest)
		if err != nil {
			return false, err
		}
		return modelReadyResponse.Ready, nil
	} else {
		requestObj := fasthttp.AcquireRequest()
		requestObj.SetRequestURI(fmt.Sprintf("%s://%s/v2/models/%s/versions/%s/ready", HTTPPrefix, t.ServerURL, modelName, modelVersion))
		responseObj := fasthttp.AcquireResponse()
		if err := t.httpClient.DoTimeout(requestObj, responseObj, timeout); err != nil {
			return false, err
		}
		defer fasthttp.ReleaseRequest(requestObj)
		defer fasthttp.ReleaseResponse(responseObj)
		if responseObj.StatusCode() != 200 {
			return false, fmt.Errorf("response code: %d", responseObj.StatusCode())
		}
		return true, nil
	}
}

// ServerMetadata Get server metadata
func (t *TritonClientService) ServerMetadata(timeout time.Duration, isGRPC bool) (*ServerMetadataResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		serverMetadataRequest := ServerMetadataRequest{}
		serverMetadataResponse, err := t.client.ServerMetadata(ctx, &serverMetadataRequest)
		if err != nil {
			return nil, err
		}
		return serverMetadataResponse, nil
	} else {
		requestObj := fasthttp.AcquireRequest()
		requestObj.SetRequestURI(fmt.Sprintf("%s://%s/v2", HTTPPrefix, t.ServerURL))
		responseObj := fasthttp.AcquireResponse()
		if err := t.httpClient.DoTimeout(requestObj, responseObj, timeout); err != nil {
			return nil, err
		}
		defer fasthttp.ReleaseRequest(requestObj)
		defer fasthttp.ReleaseResponse(responseObj)
		if responseObj.StatusCode() != 200 {
			return nil, fmt.Errorf("response code: %d", responseObj.StatusCode())
		}
		serverMetadataResponse := new(ServerMetadataResponse)
		jsonDecodeErr := json.Unmarshal(responseObj.Body(), &serverMetadataResponse)
		if jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return serverMetadataResponse, nil
	}
}

// ModelMetadataRequest Get model metadata
func (t *TritonClientService) ModelMetadataRequest(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (*ModelMetadataResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		modelMetadataRequest := ModelMetadataRequest{Name: modelName, Version: modelVersion}
		modelMetadataResponse, err := t.client.ModelMetadata(ctx, &modelMetadataRequest)
		if err != nil {
			return nil, err
		}
		return modelMetadataResponse, nil
	} else {
		requestObj := fasthttp.AcquireRequest()
		requestObj.SetRequestURI(fmt.Sprintf("%s://%s/v2/models/%s/versions/%s", HTTPPrefix, t.ServerURL, modelName, modelVersion))
		responseObj := fasthttp.AcquireResponse()
		if err := t.httpClient.DoTimeout(requestObj, responseObj, timeout); err != nil {
			return nil, err
		}
		defer fasthttp.ReleaseRequest(requestObj)
		defer fasthttp.ReleaseResponse(responseObj)
		if responseObj.StatusCode() != 200 {
			return nil, fmt.Errorf("response code: %d", responseObj.StatusCode())
		}
		modelMetadataResponse := new(ModelMetadataResponse)
		jsonDecodeErr := json.Unmarshal(responseObj.Body(), &modelMetadataResponse)
		if jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return modelMetadataResponse, nil
	}
}

// ModelIndex Get model repo index
func (t *TritonClientService) ModelIndex(isReady bool, timeout time.Duration, isGRPC bool) (*RepositoryIndexResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		repositoryIndexRequest := RepositoryIndexRequest{Ready: isReady}
		repositoryIndexResponse, err := t.client.RepositoryIndex(ctx, &repositoryIndexRequest)
		if err != nil {
			return nil, err
		}
		return repositoryIndexResponse, nil
	} else {
		requestObj := fasthttp.AcquireRequest()
		requestObj.SetRequestURI(fmt.Sprintf("%s://%s/v2/repository/index", HTTPPrefix, t.ServerURL))
		requestObj.Header.SetMethod(HttpPostMethod)
		requestObj.Header.SetContentType(JsonContentType)
		indexRequest := struct {
			Ready bool `json:"ready"`
		}{
			isReady,
		}
		reqBody, _ := json.Marshal(indexRequest)
		requestObj.SetBody(reqBody)
		responseObj := fasthttp.AcquireResponse()
		if err := t.httpClient.DoTimeout(requestObj, responseObj, timeout); err != nil {
			return nil, err
		}
		defer fasthttp.ReleaseRequest(requestObj)
		defer fasthttp.ReleaseResponse(responseObj)
		repositoryIndexResponse := new(RepositoryIndexResponse)
		// TODO Maybe Have some bug here
		jsonDecodeErr := json.Unmarshal(responseObj.Body(), &repositoryIndexResponse.Models)
		if jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return repositoryIndexResponse, nil
	}
}

// ModelConfiguration Get model configuration
func (t *TritonClientService) ModelConfiguration(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (interface{}, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		modelConfigRequest := ModelConfigRequest{Name: modelName, Version: modelVersion}
		modelConfigResponse, err := t.client.ModelConfig(ctx, &modelConfigRequest)
		if err != nil {
			return nil, err
		}
		return modelConfigResponse, nil
	} else {
		requestObj := fasthttp.AcquireRequest()
		requestObj.SetRequestURI(fmt.Sprintf("%s://%s/v2/models/%s/versions/%s/config", HTTPPrefix, t.ServerURL, modelName, modelVersion))
		responseObj := fasthttp.AcquireResponse()
		if err := t.httpClient.DoTimeout(requestObj, responseObj, timeout); err != nil {
			return nil, err
		}
		defer fasthttp.ReleaseRequest(requestObj)
		defer fasthttp.ReleaseResponse(responseObj)
		// Use *ModelConfigResponse in json.Unmarshal have some strange error
		return gjson.ParseBytes(responseObj.Body()), nil
	}
}

// ModelInferStats Get Model infer stats
func (t *TritonClientService) ModelInferStats(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (*ModelStatisticsResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		modelStatisticsRequest := ModelStatisticsRequest{Name: modelName, Version: modelVersion}
		modelStatisticsResponse, err := t.client.ModelStatistics(ctx, &modelStatisticsRequest)
		if err != nil {
			return nil, err
		}
		return modelStatisticsResponse, nil
	} else {
		requestObj := fasthttp.AcquireRequest()
		requestObj.SetRequestURI(fmt.Sprintf("%s://%s/v2/models/%s/versions/%s/stats", HTTPPrefix, t.ServerURL, modelName, modelVersion))
		responseObj := fasthttp.AcquireResponse()
		if err := t.httpClient.DoTimeout(requestObj, responseObj, timeout); err != nil {
			return nil, err
		}
		defer fasthttp.ReleaseRequest(requestObj)
		defer fasthttp.ReleaseResponse(responseObj)
		modelStatisticsResponse := new(ModelStatisticsResponse)
		jsonDecodeErr := json.Unmarshal(responseObj.Body(), &modelStatisticsResponse)
		if jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return modelStatisticsResponse, nil
	}
}

// ModelLoad Load Model
func (t *TritonClientService) ModelLoad(modelName string, timeout time.Duration, isGRPC bool) (*RepositoryModelLoadResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		loadRequest := RepositoryModelLoadRequest{ModelName: modelName}
		loadResponse, loadErr := t.client.RepositoryModelLoad(ctx, &loadRequest)
		if loadErr != nil {
			return nil, loadErr
		}
		return loadResponse, nil
	} else {
		return nil, nil
	}
}

// ModelUnload Unload model
func (t *TritonClientService) ModelUnload(modelName string, timeout time.Duration, isGRPC bool) (*RepositoryModelUnloadResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		unloadRequest := RepositoryModelUnloadRequest{ModelName: modelName}
		unloadResponse, unloadErr := t.client.RepositoryModelUnload(ctx, &unloadRequest)
		if unloadErr != nil {
			return nil, unloadErr
		}
		return unloadResponse, nil
	} else {
		return nil, nil
	}
}

// ShareMemoryStatus Get share memory / cuda memory status
func (t *TritonClientService) ShareMemoryStatus(isCUDA bool, regionName string, timeout time.Duration, isGRPC bool) (interface{}, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		if isCUDA {
			// CUDA Memory
			cudaSharedMemoryStatusRequest := CudaSharedMemoryStatusRequest{Name: regionName}
			cudaSharedMemoryStatusResponse, cudaStatusErr := t.client.CudaSharedMemoryStatus(ctx, &cudaSharedMemoryStatusRequest)
			if cudaStatusErr != nil {
				return nil, cudaStatusErr
			}
			return cudaSharedMemoryStatusResponse, nil
		} else {
			// System Memory
			systemSharedMemoryStatusRequest := SystemSharedMemoryStatusRequest{Name: regionName}
			systemSharedMemoryStatusResponse, systemStatusErr := t.client.SystemSharedMemoryStatus(ctx, &systemSharedMemoryStatusRequest)
			if systemStatusErr != nil {
				return nil, systemStatusErr
			}
			return systemSharedMemoryStatusResponse, nil
		}
	} else {
		// Acquire Request Resource
		requestObj := fasthttp.AcquireRequest()
		responseObj := fasthttp.AcquireResponse()

		// SetRequestURI
		if isCUDA {
			requestObj.SetRequestURI(fmt.Sprintf("%s://%s/v2/systemsharememory/region/%s/status", HTTPPrefix, t.ServerURL, regionName))
		} else {
			requestObj.SetRequestURI(fmt.Sprintf("%s://%s/v2/cudasharememory/region/%s/status", HTTPPrefix, t.ServerURL, regionName))
		}

		// Request
		if err := t.httpClient.DoTimeout(requestObj, responseObj, timeout); err != nil {
			return nil, err
		}
		defer fasthttp.ReleaseRequest(requestObj)
		defer fasthttp.ReleaseResponse(responseObj)

		// Parse Response
		if isCUDA {
			cudaSharedMemoryStatusResponse := new(CudaSharedMemoryStatusResponse)
			jsonDecodeErr := json.Unmarshal(responseObj.Body(), &cudaSharedMemoryStatusResponse)
			if jsonDecodeErr != nil {
				return nil, jsonDecodeErr
			}
			return cudaSharedMemoryStatusResponse, nil
		} else {
			systemSharedMemoryStatusResponse := new(SystemSharedMemoryStatusResponse)
			jsonDecodeErr := json.Unmarshal(responseObj.Body(), &systemSharedMemoryStatusResponse)
			if jsonDecodeErr != nil {
				return nil, jsonDecodeErr
			}
			return systemSharedMemoryStatusResponse, nil
		}
	}
}

// ConnectToTritonWithGRPC Create GRPC Connection
func (t *TritonClientService) ConnectToTritonWithGRPC() error {
	conn, err := grpc.Dial(t.ServerURL, grpc.WithInsecure())
	if err != nil {
		return err
	}
	t.grpcConn = conn
	t.client = NewGRPCInferenceServiceClient(conn)
	return nil
}

// DisconnectToTritonWithGRPC Disconnect GRPC Connection
func (t *TritonClientService) DisconnectToTritonWithGRPC() error {
	closeErr := t.grpcConn.Close()
	if closeErr != nil {
		return closeErr
	}
	return nil
}

// ConnectToTritonWithHTTP Create HTTP Connection
func (t *TritonClientService) ConnectToTritonWithHTTP() error {
	// HTTPClient global http client object
	t.httpClient = &fasthttp.Client{
		MaxConnsPerHost: 16384,
		ReadTimeout:     5 * time.Second,
		WriteTimeout:    5 * time.Second,
	}
	return nil
}

// DisconnectToTritonWithHTTP Disconnect HTTP Connection
func (t *TritonClientService) DisconnectToTritonWithHTTP() {
	t.httpClient.CloseIdleConnections()
	// Make GC
	t.httpClient = nil
}
