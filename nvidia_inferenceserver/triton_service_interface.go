package nvidia_inferenceserver

import (
	"errors"
	"strconv"
	"time"

	"github.com/goccy/go-json"
	"github.com/valyala/fasthttp"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

const (
	HTTPPrefix      string = "http://"
	HttpPostMethod  string = "POST"
	JsonContentType string = "application/json"

	TritonAPIForServerIsLive  string = "/v2/health/live"
	TritonAPIForServerIsReady string = "/v2/health/ready"
	TritonAPIForRepoIndex     string = "/v2/repository/index"
)

// DecoderFunc Infer Callback Function
type DecoderFunc func(response interface{}, params ...interface{}) (interface{}, error)

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
	ModelGRPCInfer(
		inferInputs []*ModelInferRequest_InferInputTensor,
		inferOutputs []*ModelInferRequest_InferRequestedOutputTensor,
		rawInputs [][]byte,
		modelName, modelVersion string,
		timeout time.Duration,
		decoderFunc DecoderFunc,
		params ...interface{},
	) (interface{}, error)
	// ModelHTTPInfer all triton inference server infer with HTTP
	ModelHTTPInfer(
		requestBody []byte,
		modelName, modelVersion string,
		timeout time.Duration,
		decoderFunc DecoderFunc, params ...interface{}) (interface{}, error)
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
	// ShareMemoryRegister Register share memory / share cuda memory).
	ShareMemoryRegister(
		isCUDA bool,
		regionName, cpuMemRegionKey string,
		cudaRawHandle []byte,
		cudaDeviceId int64,
		byteSize, cpuMemOffset uint64,
		timeout time.Duration,
		isGRPC bool) (interface{}, error)
	// ShareMemoryUnRegister Unregister share memory / share cuda memory).
	ShareMemoryUnRegister(isCUDA bool, regionName string, timeout time.Duration, isGRPC bool) (interface{}, error)
	// GetModelTracingSetting get the current trace setting
	GetModelTracingSetting(modelName string, timeout time.Duration, isGRPC bool) (*TraceSettingResponse, error)
	// SetModelTracingSetting set the current trace setting
	SetModelTracingSetting(modelName string, settingMap map[string]*TraceSettingRequest_SettingValue, timeout time.Duration, isGRPC bool) (*TraceSettingResponse, error)

	// ConnectToTritonWithGRPC : Connect to triton with GRPC
	ConnectToTritonWithGRPC() error
	// DisconnectToTritonWithGRPC : Close GRPC Connections
	DisconnectToTritonWithGRPC() error
	// ConnectToTritonWithHTTP : Create HTTP Client Pool
	ConnectToTritonWithHTTP()
	// DisconnectToTritonWithHTTP : Close HTTP Client Pool
	DisconnectToTritonWithHTTP()
}

// TritonClientService ServiceClient
type TritonClientService struct {
	ServerURL string

	grpcConn   *grpc.ClientConn
	grpcClient GRPCInferenceServiceClient

	httpClient       *fasthttp.Client
	httpRequestPool  *fasthttp.Request
	httpResponsePool *fasthttp.Response
}

// acquireOrReleaseHttpRequest
func (t *TritonClientService) acquireOrReleaseHttpRequest(isRelease bool) {
	if isRelease {
		if t.httpRequestPool != nil {
			fasthttp.ReleaseRequest(t.httpRequestPool)
		}
		return
	}
	t.httpRequestPool = fasthttp.AcquireRequest()
	t.httpRequestPool.Header.SetMethod(HttpPostMethod)
	t.httpRequestPool.Header.SetContentType(JsonContentType)
}

// acquireOrReleaseHttpResponse
func (t *TritonClientService) acquireOrReleaseHttpResponse(isRelease bool) {
	if isRelease {
		if t.httpResponsePool != nil {
			fasthttp.ReleaseResponse(t.httpResponsePool)
		}
		return
	}
	t.httpResponsePool = fasthttp.AcquireResponse()
}

// makeHttpPostRequestWithDoTimeout
func (t *TritonClientService) makeHttpPostRequestWithDoTimeout(uri string, reqBody []byte, timeout time.Duration) ([]byte, int, error) {
	if uri == "" {
		return nil, -1, nil
	}
	t.httpRequestPool.SetRequestURI(uri)
	if reqBody != nil {
		t.httpRequestPool.SetBody(reqBody)
	}
	if httpErr := t.httpClient.DoTimeout(t.httpRequestPool, t.httpResponsePool, timeout); httpErr != nil {
		return nil, t.httpResponsePool.StatusCode(), httpErr
	}
	return t.httpResponsePool.Body(), t.httpResponsePool.StatusCode(), nil
}

// makeHttpGetRequestWithDoTimeout
func (t *TritonClientService) makeHttpGetRequestWithDoTimeout(uri string, timeout time.Duration) ([]byte, int, error) {
	if uri == "" {
		return nil, -1, nil
	}
	t.httpRequestPool.SetRequestURI(uri)
	if httpErr := t.httpClient.DoTimeout(t.httpRequestPool, t.httpResponsePool, timeout); httpErr != nil {
		return nil, t.httpResponsePool.StatusCode(), httpErr
	}
	return t.httpResponsePool.Body(), t.httpResponsePool.StatusCode(), nil
}

// modelGRPCInfer Call Triton with GRPC（core function）
func (t *TritonClientService) modelGRPCInfer(
	inferInputs []*ModelInferRequest_InferInputTensor,
	inferOutputs []*ModelInferRequest_InferRequestedOutputTensor,
	rawInputs [][]byte,
	modelName, modelVersion string,
	timeout time.Duration,
) (*ModelInferResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	// Create infer request for specific model/version
	modelInferRequest := ModelInferRequest{
		ModelName:        modelName,
		ModelVersion:     modelVersion,
		Inputs:           inferInputs,
		Outputs:          inferOutputs,
		RawInputContents: rawInputs,
	}
	// Get infer response
	modelInferResponse, inferErr := t.grpcClient.ModelInfer(ctx, &modelInferRequest)
	if inferErr != nil {
		return nil, errors.New("inferErr: " + inferErr.Error())
	}
	return modelInferResponse, nil
}

///////////////////////////////////////////// expose API below /////////////////////////////////////////////

// ModelHTTPInfer Call Triton with HTTP
func (t *TritonClientService) ModelHTTPInfer(
	requestBody []byte,
	modelName, modelVersion string,
	timeout time.Duration,
	decoderFunc DecoderFunc,
	params ...interface{},
) (interface{}, error) {
	// get infer response
	modelInferResponse, modelInferStatusCode, inferErr := t.makeHttpPostRequestWithDoTimeout(
		HTTPPrefix+t.ServerURL+"/v2/models/"+modelName+"/versions/"+modelVersion+"/infer",
		requestBody,
		timeout)
	if inferErr != nil || modelInferStatusCode != fasthttp.StatusOK {
		return nil, errors.New("response code: " + strconv.Itoa(modelInferStatusCode) + "; response error: " + inferErr.Error())
	}
	// decode Result
	response, decodeErr := decoderFunc(modelInferResponse, params...)
	if decodeErr != nil {
		return nil, errors.New("decodeErr: " + decodeErr.Error())
	}
	return response, nil
}

// ModelGRPCInfer Call Triton with GRPC
func (t *TritonClientService) ModelGRPCInfer(
	inferInputs []*ModelInferRequest_InferInputTensor,
	inferOutputs []*ModelInferRequest_InferRequestedOutputTensor,
	rawInputs [][]byte,
	modelName, modelVersion string,
	timeout time.Duration,
	decoderFunc DecoderFunc,
	params ...interface{},
) (interface{}, error) {
	// Get infer response
	modelInferResponse, inferErr := t.modelGRPCInfer(inferInputs, inferOutputs, rawInputs, modelName, modelVersion, timeout)
	if inferErr != nil {
		return nil, inferErr
	}
	// decode Result
	response, decodeErr := decoderFunc(modelInferResponse, params...)
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

		serverLiveResponse, err := t.grpcClient.ServerLive(ctx, &ServerLiveRequest{})
		if err != nil {
			return false, err
		}
		return serverLiveResponse.Live, nil
	} else {
		_, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForServerIsLive, nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return false, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		return true, nil
	}
}

// CheckServerReady check server is ready
func (t *TritonClientService) CheckServerReady(timeout time.Duration, isGRPC bool) (bool, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		serverReadyResponse, err := t.grpcClient.ServerReady(ctx, &ServerReadyRequest{})
		if err != nil {
			return false, err
		}
		return serverReadyResponse.Ready, nil
	} else {
		_, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForServerIsReady, nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return false, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		return true, nil
	}
}

// CheckModelReady check model is ready
func (t *TritonClientService) CheckModelReady(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (bool, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		modelReadyResponse, err := t.grpcClient.ModelReady(ctx, &ModelReadyRequest{Name: modelName, Version: modelVersion})
		if err != nil {
			return false, err
		}
		return modelReadyResponse.Ready, nil
	} else {
		_, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+"/v2/models/"+modelName+"/versions/"+modelVersion+"/ready", nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return false, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		return true, nil
	}
}

// ServerMetadata Get server metadata
func (t *TritonClientService) ServerMetadata(timeout time.Duration, isGRPC bool) (*ServerMetadataResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		serverMetadataResponse, err := t.grpcClient.ServerMetadata(ctx, &ServerMetadataRequest{})
		if err != nil {
			return nil, err
		}
		return serverMetadataResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+"/v2", nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		serverMetadataResponse := new(ServerMetadataResponse)
		if jsonDecodeErr := json.Unmarshal(respBody, &serverMetadataResponse); jsonDecodeErr != nil {
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

		modelMetadataResponse, err := t.grpcClient.ModelMetadata(ctx, &ModelMetadataRequest{Name: modelName, Version: modelVersion})
		if err != nil {
			return nil, err
		}
		return modelMetadataResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+"/v2/models/"+modelName+"/versions/"+modelVersion, nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		modelMetadataResponse := new(ModelMetadataResponse)
		if jsonDecodeErr := json.Unmarshal(respBody, &modelMetadataResponse); jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return modelMetadataResponse, nil
	}
}

// ModelIndex Get model repo index
func (t *TritonClientService) ModelIndex(repoName string, isReady bool, timeout time.Duration, isGRPC bool) (*RepositoryIndexResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		// The name of the repository. If empty the index is returned for all repositories.
		repositoryIndexResponse, err := t.grpcClient.RepositoryIndex(ctx, &RepositoryIndexRequest{RepositoryName: repoName, Ready: isReady})
		if err != nil {
			return nil, err
		}
		return repositoryIndexResponse, nil
	} else {
		indexRequest := struct {
			RepoName string `json:"repository_name"`
			Ready    bool   `json:"ready"`
		}{
			repoName,
			isReady,
		}
		reqBody, jsonEncodeErr := json.Marshal(indexRequest)
		if jsonEncodeErr != nil {
			return nil, jsonEncodeErr
		}
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForRepoIndex, reqBody, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		repositoryIndexResponse := new(RepositoryIndexResponse)
		if jsonDecodeErr := json.Unmarshal(respBody, &repositoryIndexResponse.Models); jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return repositoryIndexResponse, nil
	}
}

// ModelConfiguration Get model configuration
func (t *TritonClientService) ModelConfiguration(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (*ModelConfigResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		modelConfigResponse, err := t.grpcClient.ModelConfig(ctx, &ModelConfigRequest{Name: modelName, Version: modelVersion})
		if err != nil {
			return nil, err
		}
		return modelConfigResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpGetRequestWithDoTimeout(HTTPPrefix+t.ServerURL+"/v2/models/"+modelName+"/versions/"+modelVersion+"/config", timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		modelConfigResponse := new(ModelConfigResponse)
		if jsonDecodeErr := json.Unmarshal(respBody, &modelConfigResponse); jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return modelConfigResponse, nil
	}
}

// ModelInferStats Get Model infer stats
func (t *TritonClientService) ModelInferStats(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (*ModelStatisticsResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		modelStatisticsResponse, err := t.grpcClient.ModelStatistics(ctx, &ModelStatisticsRequest{Name: modelName, Version: modelVersion})
		if err != nil {
			return nil, err
		}
		return modelStatisticsResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpGetRequestWithDoTimeout(HTTPPrefix+t.ServerURL+"/v2/models/"+modelName+"/versions/"+modelVersion+"/stats", timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		modelStatisticsResponse := new(ModelStatisticsResponse)
		jsonDecodeErr := json.Unmarshal(respBody, &modelStatisticsResponse)
		if jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return modelStatisticsResponse, nil
	}
}

// ModelLoad Load Model
func (t *TritonClientService) ModelLoad(repoName, modelName string, timeout time.Duration, isGRPC bool) (*RepositoryModelLoadResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		// The name of the repository to load from. If empty the model is loaded from any repository.
		loadResponse, loadErr := t.grpcClient.RepositoryModelLoad(ctx, &RepositoryModelLoadRequest{RepositoryName: repoName, ModelName: modelName})
		if loadErr != nil {
			return nil, loadErr
		}
		return loadResponse, nil
	} else {
		// https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md#examples
		// TODO hard to implements because it will take a long body to make http request
		//_, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+"/v2/repository/models/"+modelName+"/load", reqBody, timeout)
		return nil, nil
	}
}

// ModelUnload Unload model
func (t *TritonClientService) ModelUnload(repoName, modelName string, timeout time.Duration, isGRPC bool) (*RepositoryModelUnloadResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		unloadResponse, unloadErr := t.grpcClient.RepositoryModelUnload(ctx, &RepositoryModelUnloadRequest{RepositoryName: repoName, ModelName: modelName})
		if unloadErr != nil {
			return nil, unloadErr
		}
		return unloadResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+"/v2/repository/models/"+modelName+"/unload", nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		repositoryModelUnloadResponse := new(RepositoryModelUnloadResponse)
		jsonDecodeErr := json.Unmarshal(respBody, &repositoryModelUnloadResponse)
		if jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return repositoryModelUnloadResponse, nil
	}
}

// ShareMemoryStatus Get share memory / cuda memory status. Response: CudaSharedMemoryStatusResponse / SystemSharedMemoryStatusResponse
func (t *TritonClientService) ShareMemoryStatus(isCUDA bool, regionName string, timeout time.Duration, isGRPC bool) (interface{}, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		if isCUDA {
			// CUDA Memory
			cudaSharedMemoryStatusResponse, cudaStatusErr := t.grpcClient.CudaSharedMemoryStatus(ctx, &CudaSharedMemoryStatusRequest{Name: regionName})
			if cudaStatusErr != nil {
				return nil, cudaStatusErr
			}
			return cudaSharedMemoryStatusResponse, nil
		} else {
			// System Memory
			systemSharedMemoryStatusResponse, systemStatusErr := t.grpcClient.SystemSharedMemoryStatus(ctx, &SystemSharedMemoryStatusRequest{Name: regionName})
			if systemStatusErr != nil {
				return nil, systemStatusErr
			}
			return systemSharedMemoryStatusResponse, nil
		}
	} else {
		// SetRequestURI
		var uri string
		if isCUDA {
			uri = HTTPPrefix + t.ServerURL + "/v2/cudasharememory/region/" + regionName + "/status"
		} else {
			uri = HTTPPrefix + t.ServerURL + "/v2/systemsharememory/region/" + regionName + "/status"
		}
		respBody, statusCode, httpErr := t.makeHttpGetRequestWithDoTimeout(uri, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		// Parse Response
		if isCUDA {
			cudaSharedMemoryStatusResponse := new(CudaSharedMemoryStatusResponse)
			if jsonDecodeErr := json.Unmarshal(respBody, &cudaSharedMemoryStatusResponse); jsonDecodeErr != nil {
				return nil, jsonDecodeErr
			}
			return cudaSharedMemoryStatusResponse, nil
		} else {
			systemSharedMemoryStatusResponse := new(SystemSharedMemoryStatusResponse)
			if jsonDecodeErr := json.Unmarshal(respBody, &systemSharedMemoryStatusResponse); jsonDecodeErr != nil {
				return nil, jsonDecodeErr
			}
			return systemSharedMemoryStatusResponse, nil
		}
	}
}

// ShareMemoryRegister Register share memory / cuda memory
func (t *TritonClientService) ShareMemoryRegister(
	isCUDA bool,
	regionName, cpuMemRegionKey string,
	cudaRawHandle []byte,
	cudaDeviceId int64,
	byteSize, cpuMemOffset uint64,
	timeout time.Duration,
	isGRPC bool) (interface{}, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		if isCUDA {
			// CUDA Memory
			cudaSharedMemoryRegisterResponse, registerErr := t.grpcClient.CudaSharedMemoryRegister(ctx, &CudaSharedMemoryRegisterRequest{
				Name:      regionName,
				RawHandle: cudaRawHandle,
				DeviceId:  cudaDeviceId,
				ByteSize:  byteSize,
			})
			if registerErr != nil {
				return nil, registerErr
			}
			return cudaSharedMemoryRegisterResponse, nil
		} else {
			// System Memory
			systemSharedMemoryRegisterResponse, registerErr := t.grpcClient.SystemSharedMemoryRegister(ctx, &SystemSharedMemoryRegisterRequest{
				Name:     regionName,
				Key:      cpuMemRegionKey,
				Offset:   cpuMemOffset,
				ByteSize: byteSize,
			})
			if registerErr != nil {
				return nil, registerErr
			}
			return systemSharedMemoryRegisterResponse, nil
		}
	} else {
		// SetRequestURI (Experimental !!!)
		var uri string
		var reqBody []byte
		var jsonEncodeErr error
		if isCUDA {
			uri = HTTPPrefix + t.ServerURL + "/v2/cudasharememory/region/" + regionName + "/register"
			cudaMemoryRegisterBody := struct {
				RawHandle interface{} `json:"raw_handle"`
				DeviceId  int64       `json:"device_id"`
				ByteSize  uint64      `json:"byte_size"`
			}{
				cudaRawHandle,
				cudaDeviceId,
				byteSize,
			}
			reqBody, jsonEncodeErr = json.Marshal(cudaMemoryRegisterBody)
			if jsonEncodeErr != nil {
				return nil, jsonEncodeErr
			}
		} else {
			uri = HTTPPrefix + t.ServerURL + "/v2/systemsharememory/region/" + regionName + "/register"
			systemMemoryRegisterBody := struct {
				Key      string `json:"key"`
				Offset   uint64 `json:"offset"`
				ByteSize uint64 `json:"byte_size"`
			}{
				cpuMemRegionKey,
				cpuMemOffset,
				byteSize,
			}
			reqBody, jsonEncodeErr = json.Marshal(systemMemoryRegisterBody)
			if jsonEncodeErr != nil {
				return nil, jsonEncodeErr
			}
		}
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(uri, reqBody, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		// Parse Response
		if isCUDA {
			cudaSharedMemoryRegisterResponse := new(CudaSharedMemoryRegisterResponse)
			if jsonDecodeErr := json.Unmarshal(respBody, &cudaSharedMemoryRegisterResponse); jsonDecodeErr != nil {
				return nil, jsonDecodeErr
			}
			return cudaSharedMemoryRegisterResponse, nil
		} else {
			systemSharedMemoryRegisterResponse := new(SystemSharedMemoryRegisterResponse)
			if jsonDecodeErr := json.Unmarshal(respBody, &systemSharedMemoryRegisterResponse); jsonDecodeErr != nil {
				return nil, jsonDecodeErr
			}
			return systemSharedMemoryRegisterResponse, nil
		}
	}
}

func (t *TritonClientService) ShareMemoryUnRegister(isCUDA bool, regionName string, timeout time.Duration, isGRPC bool) (interface{}, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		if isCUDA {
			// CUDA Memory
			cudaSharedMemoryUnRegisterResponse, unRegisterErr := t.grpcClient.CudaSharedMemoryUnregister(ctx, &CudaSharedMemoryUnregisterRequest{Name: regionName})
			if unRegisterErr != nil {
				return nil, unRegisterErr
			}
			return cudaSharedMemoryUnRegisterResponse, nil
		} else {
			// System Memory
			systemSharedMemoryUnRegisterResponse, unRegisterErr := t.grpcClient.SystemSharedMemoryUnregister(ctx, &SystemSharedMemoryUnregisterRequest{Name: regionName})
			if unRegisterErr != nil {
				return nil, unRegisterErr
			}
			return systemSharedMemoryUnRegisterResponse, nil
		}
	} else {
		// SetRequestURI
		var uri string
		if isCUDA {
			uri = HTTPPrefix + t.ServerURL + "/v2/cudasharememory/region/" + regionName + "/unregister"
		} else {
			uri = HTTPPrefix + t.ServerURL + "/v2/systemsharememory/region/" + regionName + "/unregister"
		}
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(uri, nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		// Parse Response
		if isCUDA {
			cudaSharedMemoryUnregisterResponse := new(CudaSharedMemoryUnregisterResponse)
			if jsonDecodeErr := json.Unmarshal(respBody, &cudaSharedMemoryUnregisterResponse); jsonDecodeErr != nil {
				return nil, jsonDecodeErr
			}
			return cudaSharedMemoryUnregisterResponse, nil
		} else {
			systemSharedMemoryUnregisterResponse := new(SystemSharedMemoryUnregisterResponse)
			if jsonDecodeErr := json.Unmarshal(respBody, &systemSharedMemoryUnregisterResponse); jsonDecodeErr != nil {
				return nil, jsonDecodeErr
			}
			return systemSharedMemoryUnregisterResponse, nil
		}
	}
}

// GetModelTracingSetting get model tracing setting
func (t *TritonClientService) GetModelTracingSetting(modelName string, timeout time.Duration, isGRPC bool) (*TraceSettingResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		traceSettingResponse, err := t.grpcClient.TraceSetting(ctx, &TraceSettingRequest{ModelName: modelName})
		if err != nil {
			return nil, err
		}
		return traceSettingResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpGetRequestWithDoTimeout(HTTPPrefix+t.ServerURL+"/v2/models/"+modelName+"/trace/setting", timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		traceSettingResponse := new(TraceSettingResponse)
		if jsonDecodeErr := json.Unmarshal(respBody, traceSettingResponse); jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return traceSettingResponse, nil
	}
}

// SetModelTracingSetting set model tracing setting
// Param: settingMap ==> https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_trace.md#trace-setting-response-json-object
func (t *TritonClientService) SetModelTracingSetting(modelName string, settingMap map[string]*TraceSettingRequest_SettingValue, timeout time.Duration, isGRPC bool) (*TraceSettingResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		traceSettingResponse, err := t.grpcClient.TraceSetting(ctx, &TraceSettingRequest{ModelName: modelName, Settings: settingMap})
		if err != nil {
			return nil, err
		}
		return traceSettingResponse, nil
	} else {
		// Experimental
		traceSettingRequest := struct {
			TraceSetting interface{} `json:"trace_setting"`
		}{
			settingMap,
		}
		reqBody, jsonEncodeErr := json.Marshal(traceSettingRequest)
		if jsonEncodeErr != nil {
			return nil, jsonEncodeErr
		}
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+"/v2/models/"+modelName+"/trace/setting", reqBody, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, errors.New("response code: " + strconv.Itoa(statusCode) + "; response error: " + httpErr.Error())
		}
		traceSettingResponse := new(TraceSettingResponse)
		if jsonDecodeErr := json.Unmarshal(respBody, traceSettingResponse); jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return traceSettingResponse, nil
	}
}

// ConnectToTritonWithGRPC Create GRPC Connection
func (t *TritonClientService) ConnectToTritonWithGRPC() error {
	conn, err := grpc.Dial(t.ServerURL, grpc.WithInsecure())
	if err != nil {
		return err
	}
	t.grpcConn = conn
	t.grpcClient = NewGRPCInferenceServiceClient(conn)
	return nil
}

// DisconnectToTritonWithGRPC Disconnect GRPC Connection
func (t *TritonClientService) DisconnectToTritonWithGRPC() error {
	if closeErr := t.grpcConn.Close(); closeErr != nil {
		return closeErr
	}
	return nil
}

// ConnectToTritonWithHTTP Create HTTP Connection
func (t *TritonClientService) ConnectToTritonWithHTTP(client *fasthttp.Client) error {
	if client == nil {
		// HTTPClient global http client object
		t.httpClient = &fasthttp.Client{
			MaxConnsPerHost: 16384,
			ReadTimeout:     5 * time.Second,
			WriteTimeout:    5 * time.Second,
		}
		return nil
	}
	t.httpClient = client
	return nil
}

// DisconnectToTritonWithHTTP Disconnect HTTP Connection
func (t *TritonClientService) DisconnectToTritonWithHTTP() {
	t.httpClient.CloseIdleConnections()
	// Make GC
	t.httpClient = nil
}
