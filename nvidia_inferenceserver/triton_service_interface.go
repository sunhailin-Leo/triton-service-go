package nvidia_inferenceserver

import (
	"errors"
	"strconv"
	"time"

	"github.com/goccy/go-json"
	"github.com/valyala/fasthttp"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	HTTPPrefix                           string = "http://"
	HttpPostMethod                       string = "POST"
	JsonContentType                      string = "application/json"
	TritonAPIForModelVersionPrefix       string = "/versions/"
	TritonAPIPrefix                      string = "/v2"
	TritonAPIForServerIsLive                    = TritonAPIPrefix + "/health/live"
	TritonAPIForServerIsReady                   = TritonAPIPrefix + "/health/ready"
	TritonAPIForRepoIndex                       = TritonAPIPrefix + "/repository/index"
	TritonAPIForRepoModelPrefix                 = TritonAPIPrefix + "/repository/models/"
	TritonAPIForModelPrefix                     = TritonAPIPrefix + "/models/"
	TritonAPIForCudaMemoryRegionPrefix          = TritonAPIPrefix + "/cudasharememory/region/"
	TritonAPIForSystemMemoryRegionPrefix        = TritonAPIPrefix + "/systemsharememory/region/"
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
	httpClient *fasthttp.Client
}

// setGRPCConnection Create GRPC Connection
func (t *TritonClientService) setGRPCConnection(conn *grpc.ClientConn) error {
	if conn == nil {
		defaultConn, err := grpc.Dial(t.ServerURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			return err
		}
		t.grpcConn = defaultConn
	} else {
		t.grpcConn = conn
	}
	// use t.grpcConn
	t.grpcClient = NewGRPCInferenceServiceClient(t.grpcConn)
	return nil
}

// disconnectToTritonWithGRPC Disconnect GRPC Connection
func (t *TritonClientService) disconnectToTritonWithGRPC() error {
	if closeErr := t.grpcConn.Close(); closeErr != nil {
		return closeErr
	}
	return nil
}

// setHTTPConnection Create HTTP Connection
func (t *TritonClientService) setHTTPConnection(client *fasthttp.Client) error {
	if client == nil {
		// HTTPClient global http client object
		// Hard Code for client configuration
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

// disconnectToTritonWithHTTP Disconnect HTTP Connection
func (t *TritonClientService) disconnectToTritonWithHTTP() error {
	t.httpClient.CloseIdleConnections()
	t.httpClient = nil
	return nil
}

// acquireHttpRequest
func (t *TritonClientService) acquireHttpRequest() *fasthttp.Request {
	httpRequestPool := fasthttp.AcquireRequest()
	httpRequestPool.Header.SetMethod(HttpPostMethod)
	httpRequestPool.Header.SetContentType(JsonContentType)
	return httpRequestPool
}

// releaseHttpRequest
func (t *TritonClientService) releaseHttpRequest(requestObj *fasthttp.Request) {
	fasthttp.ReleaseRequest(requestObj)
}

// acquireHttpResponse
func (t *TritonClientService) acquireHttpResponse() *fasthttp.Response {
	return fasthttp.AcquireResponse()
}

// releaseHttpResponse
func (t *TritonClientService) releaseHttpResponse(responseObj *fasthttp.Response) {
	fasthttp.ReleaseResponse(responseObj)
}

// makeHttpPostRequestWithDoTimeout
func (t *TritonClientService) makeHttpPostRequestWithDoTimeout(uri string, reqBody []byte, timeout time.Duration) ([]byte, int, error) {
	if uri == "" {
		return nil, -1, nil
	}
	requestObj := t.acquireHttpRequest()
	responseObj := t.acquireHttpResponse()
	defer func() {
		t.releaseHttpRequest(requestObj)
		t.releaseHttpResponse(responseObj)
	}()
	requestObj.SetRequestURI(uri)
	if reqBody != nil {
		requestObj.SetBody(reqBody)
	}
	if httpErr := t.httpClient.DoTimeout(requestObj, responseObj, timeout); httpErr != nil {
		return nil, responseObj.StatusCode(), httpErr
	}
	return responseObj.Body(), responseObj.StatusCode(), nil
}

// makeHttpGetRequestWithDoTimeout
func (t *TritonClientService) makeHttpGetRequestWithDoTimeout(uri string, timeout time.Duration) ([]byte, int, error) {
	if uri == "" {
		return nil, -1, nil
	}
	requestObj := t.acquireHttpRequest()
	responseObj := t.acquireHttpResponse()
	defer func() {
		t.releaseHttpRequest(requestObj)
		t.releaseHttpResponse(responseObj)
	}()
	requestObj.SetRequestURI(uri)
	if httpErr := t.httpClient.DoTimeout(requestObj, responseObj, timeout); httpErr != nil {
		return nil, responseObj.StatusCode(), httpErr
	}
	return responseObj.Body(), responseObj.StatusCode(), nil
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

// httpErrorHandler HTTP Error Handler
func (t *TritonClientService) httpErrorHandler(statusCode int, httpErr error) error {
	return errors.New("[HTTP]code: " + strconv.Itoa(statusCode) + "; error: " + httpErr.Error())
}

// grpcErrorHandler GRPC Error Handler
func (t *TritonClientService) grpcErrorHandler(grpcErr error) error {
	return errors.New("[GRPC]error: " + grpcErr.Error())
}

// decodeFuncErrorHandler DecodeFunc Error Handler
func (t *TritonClientService) decodeFuncErrorHandler(err error, isGRPC bool) error {
	if isGRPC {
		return errors.New("[GRPC]decodeFunc error: " + err.Error())
	}
	return errors.New("[HTTP]decodeFunc error: " + err.Error())
}

///////////////////////////////////////////// expose API below /////////////////////////////////////////////

// ModelHTTPInfer Call Triton Infer with HTTP
func (t *TritonClientService) ModelHTTPInfer(
	requestBody []byte,
	modelName, modelVersion string,
	timeout time.Duration,
	decoderFunc DecoderFunc,
	params ...interface{},
) (interface{}, error) {
	// get infer response
	modelInferResponse, modelInferStatusCode, inferErr := t.makeHttpPostRequestWithDoTimeout(
		HTTPPrefix+t.ServerURL+TritonAPIForModelPrefix+modelName+TritonAPIForModelVersionPrefix+modelVersion+"/infer",
		requestBody,
		timeout)
	if inferErr != nil || modelInferStatusCode != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(modelInferStatusCode, inferErr)
	}
	// decode Result
	response, decodeErr := decoderFunc(modelInferResponse, params...)
	if decodeErr != nil {
		return nil, t.decodeFuncErrorHandler(decodeErr, false)
	}
	return response, nil
}

// ModelGRPCInfer Call Triton Infer with GRPC
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
		return nil, t.grpcErrorHandler(inferErr)
	}
	// decode Result
	response, decodeErr := decoderFunc(modelInferResponse, params...)
	if decodeErr != nil {
		return nil, t.decodeFuncErrorHandler(decodeErr, true)
	}
	return response, nil
}

// CheckServerAlive check server is alive
func (t *TritonClientService) CheckServerAlive(timeout time.Duration, isGRPC bool) (bool, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		serverLiveResponse, serverAliveErr := t.grpcClient.ServerLive(ctx, &ServerLiveRequest{})
		if serverAliveErr != nil {
			return false, t.grpcErrorHandler(serverAliveErr)
		}
		return serverLiveResponse.Live, nil
	} else {
		_, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForServerIsLive, nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return false, t.httpErrorHandler(statusCode, httpErr)
		}
		return true, nil
	}
}

// CheckServerReady check server is ready
func (t *TritonClientService) CheckServerReady(timeout time.Duration, isGRPC bool) (bool, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		serverReadyResponse, serverReadyErr := t.grpcClient.ServerReady(ctx, &ServerReadyRequest{})
		if serverReadyErr != nil {
			return false, t.grpcErrorHandler(serverReadyErr)
		}
		return serverReadyResponse.Ready, nil
	} else {
		_, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForServerIsReady, nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return false, t.httpErrorHandler(statusCode, httpErr)
		}
		return true, nil
	}
}

// CheckModelReady check model is ready
func (t *TritonClientService) CheckModelReady(modelName, modelVersion string, timeout time.Duration, isGRPC bool) (bool, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		modelReadyResponse, modelReadyErr := t.grpcClient.ModelReady(ctx, &ModelReadyRequest{Name: modelName, Version: modelVersion})
		if modelReadyErr != nil {
			return false, t.grpcErrorHandler(modelReadyErr)
		}
		return modelReadyResponse.Ready, nil
	} else {
		_, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForModelPrefix+modelName+TritonAPIForModelVersionPrefix+modelVersion+"/ready", nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return false, t.httpErrorHandler(statusCode, httpErr)
		}
		return true, nil
	}
}

// ServerMetadata Get server metadata
func (t *TritonClientService) ServerMetadata(timeout time.Duration, isGRPC bool) (*ServerMetadataResponse, error) {
	if isGRPC {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		serverMetadataResponse, serverMetaErr := t.grpcClient.ServerMetadata(ctx, &ServerMetadataRequest{})
		if serverMetaErr != nil {
			return nil, t.grpcErrorHandler(serverMetaErr)
		}
		return serverMetadataResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIPrefix, nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
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

		modelMetadataResponse, modelMetaErr := t.grpcClient.ModelMetadata(ctx, &ModelMetadataRequest{Name: modelName, Version: modelVersion})
		if modelMetaErr != nil {
			return nil, t.grpcErrorHandler(modelMetaErr)
		}
		return modelMetadataResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForModelPrefix+modelName+TritonAPIForModelVersionPrefix+modelVersion, nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
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
		repositoryIndexResponse, modelIndexErr := t.grpcClient.RepositoryIndex(ctx, &RepositoryIndexRequest{RepositoryName: repoName, Ready: isReady})
		if modelIndexErr != nil {
			return nil, t.grpcErrorHandler(modelIndexErr)
		}
		return repositoryIndexResponse, nil
	} else {
		reqBody, jsonEncodeErr := json.Marshal(&ModelIndexRequestHTTPObj{repoName, isReady})
		if jsonEncodeErr != nil {
			return nil, jsonEncodeErr
		}
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForRepoIndex, reqBody, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
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

		modelConfigResponse, getModelConfigErr := t.grpcClient.ModelConfig(ctx, &ModelConfigRequest{Name: modelName, Version: modelVersion})
		if getModelConfigErr != nil {
			return nil, t.grpcErrorHandler(getModelConfigErr)
		}
		return modelConfigResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpGetRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForModelPrefix+modelName+TritonAPIForModelVersionPrefix+modelVersion+"/config", timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
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

		modelStatisticsResponse, getInferStatsErr := t.grpcClient.ModelStatistics(ctx, &ModelStatisticsRequest{Name: modelName, Version: modelVersion})
		if getInferStatsErr != nil {
			return nil, t.grpcErrorHandler(getInferStatsErr)
		}
		return modelStatisticsResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpGetRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForModelPrefix+modelName+TritonAPIForModelVersionPrefix+modelVersion+"/stats", timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
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
			return nil, t.grpcErrorHandler(loadErr)
		}
		return loadResponse, nil
	} else {
		// https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md#examples
		// TODO hard to implements because it will take a long body to make http request
		//_, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForRepoModelPrefix+modelName+"/load", reqBody, timeout)
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
			return nil, t.grpcErrorHandler(unloadErr)
		}
		return unloadResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForRepoModelPrefix+modelName+"/unload", nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
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
				return nil, t.grpcErrorHandler(cudaStatusErr)
			}
			return cudaSharedMemoryStatusResponse, nil
		} else {
			// System Memory
			systemSharedMemoryStatusResponse, systemStatusErr := t.grpcClient.SystemSharedMemoryStatus(ctx, &SystemSharedMemoryStatusRequest{Name: regionName})
			if systemStatusErr != nil {
				return nil, t.grpcErrorHandler(systemStatusErr)
			}
			return systemSharedMemoryStatusResponse, nil
		}
	} else {
		// SetRequestURI
		var uri string
		if isCUDA {
			uri = HTTPPrefix + t.ServerURL + TritonAPIForCudaMemoryRegionPrefix + regionName + "/status"
		} else {
			uri = HTTPPrefix + t.ServerURL + TritonAPIForSystemMemoryRegionPrefix + regionName + "/status"
		}
		respBody, statusCode, httpErr := t.makeHttpGetRequestWithDoTimeout(uri, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
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
				return nil, t.grpcErrorHandler(registerErr)
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
				return nil, t.grpcErrorHandler(registerErr)
			}
			return systemSharedMemoryRegisterResponse, nil
		}
	} else {
		// SetRequestURI (Experimental !!!)
		var uri string
		var reqBody []byte
		var jsonEncodeErr error
		if isCUDA {
			uri = HTTPPrefix + t.ServerURL + TritonAPIForCudaMemoryRegionPrefix + regionName + "/register"
			reqBody, jsonEncodeErr = json.Marshal(&CudaMemoryRegisterBodyHTTPObj{cudaRawHandle, cudaDeviceId, byteSize})
			if jsonEncodeErr != nil {
				return nil, jsonEncodeErr
			}
		} else {
			uri = HTTPPrefix + t.ServerURL + TritonAPIForSystemMemoryRegionPrefix + regionName + "/register"
			reqBody, jsonEncodeErr = json.Marshal(&SystemMemoryRegisterBodyHTTPObj{cpuMemRegionKey, cpuMemOffset, byteSize})
			if jsonEncodeErr != nil {
				return nil, jsonEncodeErr
			}
		}
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(uri, reqBody, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
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
				return nil, t.grpcErrorHandler(unRegisterErr)
			}
			return cudaSharedMemoryUnRegisterResponse, nil
		} else {
			// System Memory
			systemSharedMemoryUnRegisterResponse, unRegisterErr := t.grpcClient.SystemSharedMemoryUnregister(ctx, &SystemSharedMemoryUnregisterRequest{Name: regionName})
			if unRegisterErr != nil {
				return nil, t.grpcErrorHandler(unRegisterErr)
			}
			return systemSharedMemoryUnRegisterResponse, nil
		}
	} else {
		// SetRequestURI
		var uri string
		if isCUDA {
			uri = HTTPPrefix + t.ServerURL + TritonAPIForCudaMemoryRegionPrefix + regionName + "/unregister"
		} else {
			uri = HTTPPrefix + t.ServerURL + TritonAPIForSystemMemoryRegionPrefix + regionName + "/unregister"
		}
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(uri, nil, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
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

		traceSettingResponse, getTraceSettingErr := t.grpcClient.TraceSetting(ctx, &TraceSettingRequest{ModelName: modelName})
		if getTraceSettingErr != nil {
			return nil, t.grpcErrorHandler(getTraceSettingErr)
		}
		return traceSettingResponse, nil
	} else {
		respBody, statusCode, httpErr := t.makeHttpGetRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForModelPrefix+modelName+"/trace/setting", timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
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

		traceSettingResponse, setTraceSettingErr := t.grpcClient.TraceSetting(ctx, &TraceSettingRequest{ModelName: modelName, Settings: settingMap})
		if setTraceSettingErr != nil {
			return nil, t.grpcErrorHandler(setTraceSettingErr)
		}
		return traceSettingResponse, nil
	} else {
		// Experimental
		reqBody, jsonEncodeErr := json.Marshal(&TraceSettingRequestHTTPObj{settingMap})
		if jsonEncodeErr != nil {
			return nil, jsonEncodeErr
		}
		respBody, statusCode, httpErr := t.makeHttpPostRequestWithDoTimeout(HTTPPrefix+t.ServerURL+TritonAPIForModelPrefix+modelName+"/trace/setting", reqBody, timeout)
		if httpErr != nil || statusCode != fasthttp.StatusOK {
			return nil, t.httpErrorHandler(statusCode, httpErr)
		}
		traceSettingResponse := new(TraceSettingResponse)
		if jsonDecodeErr := json.Unmarshal(respBody, traceSettingResponse); jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return traceSettingResponse, nil
	}
}

// InitTritonConnection init http and grpc connection
// support kafka endpoint in the future (When after Triton Add-on)
func (t *TritonClientService) InitTritonConnection(
	grpcClient *grpc.ClientConn,
	httpClient *fasthttp.Client,
) (connectionErr error) {
	if grpcClient != nil {
		connectionErr = t.setGRPCConnection(grpcClient)
	}
	if httpClient != nil {
		connectionErr = t.setHTTPConnection(httpClient)
	}
	if connectionErr != nil {
		return errors.New("[Triton]InitConnectionError: " + connectionErr.Error())
	}
	return nil
}

// ShutdownTritonConnection shutdown http and grpc connection
func (t *TritonClientService) ShutdownTritonConnection() (disconnectionErr error) {
	if t.grpcConn != nil {
		disconnectionErr = t.disconnectToTritonWithGRPC()
	}
	if t.httpClient != nil {
		disconnectionErr = t.disconnectToTritonWithHTTP()
	}
	if disconnectionErr != nil {
		return errors.New("[Triton]DisconnectionError: " + disconnectionErr.Error())
	}
	return nil
}
