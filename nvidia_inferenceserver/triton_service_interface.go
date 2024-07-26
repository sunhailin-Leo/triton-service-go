package nvidia_inferenceserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
)

const (
	DefaultHTTPClientReadTimeout                = 5 * time.Second
	DefaultHTTPClientWriteTimeout               = 5 * time.Second
	DefaultHTTPClientMaxConnPerHost      int    = 16384
	HTTPPrefix                           string = "http://"
	JSONContentType                      string = "application/json"
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

// DecoderFunc Infer Callback Function.
type DecoderFunc func(response interface{}, params ...interface{}) ([]interface{}, error)

// TritonService Service interface.
type TritonService interface {
	// JsonMarshal Json Encoder
	JsonMarshal(v interface{}) ([]byte, error)
	// JsonUnmarshal Json Decoder
	JsonUnmarshal(data []byte, v interface{}) error

	// CheckServerAlive Check triton inference server is alive.
	CheckServerAlive() (bool, error)
	// CheckServerReady Check triton inference server is ready.
	CheckServerReady() (bool, error)
	// CheckModelReady Check triton inference server`s model is ready.
	CheckModelReady(modelName, modelVersion string) (bool, error)
	// ServerMetadata Get triton inference server metadata.
	ServerMetadata() (*ServerMetadataResponse, error)
	// ModelGRPCInfer Call triton inference server infer with GRPC
	ModelGRPCInfer(
		inferInputs []*ModelInferRequest_InferInputTensor,
		inferOutputs []*ModelInferRequest_InferRequestedOutputTensor,
		rawInputs [][]byte,
		modelName, modelVersion string,
		decoderFunc DecoderFunc,
		params ...interface{},
	) ([]interface{}, error)
	// ModelHTTPInfer all triton inference server infer with HTTP
	ModelHTTPInfer(
		requestBody []byte,
		modelName, modelVersion string,
		decoderFunc DecoderFunc,
		params ...interface{},
	) ([]interface{}, error)
	// ModelMetadataRequest Get triton inference server`s model metadata.
	ModelMetadataRequest(modelName, modelVersion string) (*ModelMetadataResponse, error)
	// ModelIndex Get triton inference server model index.
	ModelIndex(isReady bool) (*RepositoryIndexResponse, error)
	// ModelConfiguration Get triton inference server model configuration.
	ModelConfiguration(modelName, modelVersion string) (interface{}, error)
	// ModelInferStats Get triton inference server model infer stats.
	ModelInferStats(modelName, modelVersion string) (*ModelStatisticsResponse, error)
	// ModelLoadWithHTTP Load model with http.
	ModelLoadWithHTTP(modelName string, modelConfigBody []byte) (*RepositoryModelLoadResponse, error)
	// ModelLoadWithGRPC Load model with http.
	ModelLoadWithGRPC(repoName, modelName string, modelConfigBody map[string]*ModelRepositoryParameter) (*RepositoryModelLoadResponse, error)
	// ModelUnloadWithHTTP Unload model with http.
	ModelUnloadWithHTTP(modelName string, modelConfigBody []byte) (*RepositoryModelUnloadResponse, error)
	// ModelUnloadWithGRPC Unload model with grpc.
	ModelUnloadWithGRPC(repoName, modelName string, modelConfigBody map[string]*ModelRepositoryParameter) (*RepositoryModelUnloadResponse, error)
	// ShareMemoryStatus Show share memory / share cuda memory status.
	ShareMemoryStatus(isCUDA bool, regionName string) (interface{}, error)
	// ShareCUDAMemoryRegister Register share cuda memory.
	ShareCUDAMemoryRegister(regionName string, cudaRawHandle []byte, cudaDeviceID int64, byteSize uint64) (interface{}, error)
	// ShareCUDAMemoryUnRegister Unregister share cuda memory
	ShareCUDAMemoryUnRegister(regionName string) (interface{}, error)
	// ShareSystemMemoryRegister Register system share memory.
	ShareSystemMemoryRegister(regionName, cpuMemRegionKey string, byteSize, cpuMemOffset uint64) (interface{}, error)
	// ShareSystemMemoryUnRegister Unregister system share memory.
	ShareSystemMemoryUnRegister(regionName string) (interface{}, error)
	// GetModelTracingSetting get the current trace setting.
	GetModelTracingSetting(modelName string) (*TraceSettingResponse, error)
	// SetModelTracingSetting set the current trace setting.
	SetModelTracingSetting(modelName string, settingMap map[string]*TraceSettingRequest_SettingValue) (*TraceSettingResponse, error)

	// SetSecondaryServerURL Set secondary server url
	SetSecondaryServerURL(url string)
	// SetAPIRequestTimeout Set API request timeout.
	SetAPIRequestTimeout(timeout time.Duration)
	// ShutdownTritonConnection close client connection.
	ShutdownTritonConnection() (disconnectionErr error)
}

// TraceDurationObj unit is nanoseconds.
type TraceDurationObj struct {
	PreProcessNanoDuration  int64
	InferNanoDuration       int64
	PostProcessNanoDuration int64
}

// TritonClientService ServiceClient.
type TritonClientService struct {
	serverURL          string
	secondaryServerURL string

	grpcConn   *grpc.ClientConn
	grpcClient GRPCInferenceServiceClient
	httpClient *fasthttp.Client
	apiTimeout time.Duration

	// Default: json.Marshal
	JSONEncoder utils.JSONMarshal
	// Default: json.Unmarshal
	JSONDecoder utils.JSONUnmarshal
}

// disconnectToTritonWithGRPC Disconnect GRPC Connection.
func (t *TritonClientService) disconnectToTritonWithGRPC() error {
	return t.grpcConn.Close()
}

// setHTTPConnection Create HTTP Connection.
func (t *TritonClientService) setHTTPConnection(client *fasthttp.Client) {
	if client == nil {
		// HTTPClient global http client object
		// Hard Code for client configuration
		t.httpClient = &fasthttp.Client{
			MaxConnsPerHost: DefaultHTTPClientMaxConnPerHost,
			ReadTimeout:     DefaultHTTPClientReadTimeout,
			WriteTimeout:    DefaultHTTPClientWriteTimeout,
		}
	}
	t.httpClient = client
}

// disconnectToTritonWithHTTP Disconnect HTTP Connection.
func (t *TritonClientService) disconnectToTritonWithHTTP() {
	t.httpClient.CloseIdleConnections()
	t.httpClient = nil
}

// acquireHTTPRequest acquire http request.
func (t *TritonClientService) acquireHTTPRequest(method string) *fasthttp.Request {
	httpRequestPool := fasthttp.AcquireRequest()
	httpRequestPool.Header.SetMethod(method)
	httpRequestPool.Header.SetContentType(JSONContentType)
	return httpRequestPool
}

// getServerURL get server url. If secondaryServerURL is not empty, this function was return it.
func (t *TritonClientService) getServerURL() string {
	if t.secondaryServerURL == "" {
		return HTTPPrefix + t.serverURL
	}
	return HTTPPrefix + t.secondaryServerURL
}

// makeHTTPPostRequestWithDoTimeout make http post request with timeout.
func (t *TritonClientService) makeHTTPPostRequestWithDoTimeout(uri string, reqBody []byte) (*fasthttp.Response, error) {
	requestObj := t.acquireHTTPRequest(fasthttp.MethodPost)
	responseObj := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(requestObj)

	requestObj.SetRequestURI(uri)
	if reqBody != nil {
		requestObj.SetBody(reqBody)
	}
	if httpErr := t.httpClient.DoTimeout(requestObj, responseObj, t.apiTimeout); httpErr != nil {
		return responseObj, httpErr
	}
	return responseObj, nil
}

// makeHTTPGetRequestWithDoTimeout make http get request with timeout.
func (t *TritonClientService) makeHTTPGetRequestWithDoTimeout(uri string) (*fasthttp.Response, error) {
	requestObj := t.acquireHTTPRequest(fasthttp.MethodGet)
	responseObj := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(requestObj)

	requestObj.SetRequestURI(uri)
	if httpErr := t.httpClient.DoTimeout(requestObj, responseObj, t.apiTimeout); httpErr != nil {
		return responseObj, httpErr
	}
	return responseObj, nil
}

// modelGRPCInfer Call Triton with GRPC（core function）.
func (t *TritonClientService) modelGRPCInfer(
	inferInputs []*ModelInferRequest_InferInputTensor,
	inferOutputs []*ModelInferRequest_InferRequestedOutputTensor,
	rawInputs [][]byte,
	modelName, modelVersion string,
) (*ModelInferResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
	defer cancel()
	// Create infer request for specific model/version.
	modelInferRequest := ModelInferRequest{
		ModelName:        modelName,
		ModelVersion:     modelVersion,
		Inputs:           inferInputs,
		Outputs:          inferOutputs,
		RawInputContents: rawInputs,
	}
	// Get infer response.
	modelInferResponse, inferErr := t.grpcClient.ModelInfer(ctx, &modelInferRequest)
	if inferErr != nil {
		return nil, errors.New("[GRPC]inferErr: " + inferErr.Error())
	}
	return modelInferResponse, nil
}

// httpErrorHandler HTTP Error Handler.
func (t *TritonClientService) httpErrorHandler(statusCode int, httpErr error) error {
	if httpErr != nil {
		return errors.New("[HTTP]code: " + strconv.Itoa(statusCode) + "; error: " + httpErr.Error())
	}
	return nil
}

// grpcErrorHandler GRPC Error Handler.
func (t *TritonClientService) grpcErrorHandler(grpcErr error) error {
	if grpcErr != nil {
		return errors.New("[GRPC]error: " + grpcErr.Error())
	}
	return nil
}

// decodeFuncErrorHandler DecodeFunc Error Handler.
func (t *TritonClientService) decodeFuncErrorHandler(err error, isGRPC bool) error {
	if isGRPC {
		return errors.New("[GRPC]decodeFunc error: " + err.Error())
	}
	return errors.New("[HTTP]decodeFunc error: " + err.Error())
}

///////////////////////////////////////////// expose API below /////////////////////////////////////////////

// JsonMarshal Json Encoder
func (t *TritonClientService) JsonMarshal(v interface{}) ([]byte, error) {
	return t.JSONEncoder(v)
}

// JsonUnmarshal Json Decoder
func (t *TritonClientService) JsonUnmarshal(data []byte, v interface{}) error {
	return t.JSONDecoder(data, v)
}

// SetJSONEncoder set json encoder
func (t *TritonClientService) SetJSONEncoder(encoder utils.JSONMarshal) *TritonClientService {
	t.JSONEncoder = encoder
	return t
}

// SetJsonDecoder set json decoder
func (t *TritonClientService) SetJsonDecoder(decoder utils.JSONUnmarshal) *TritonClientService {
	t.JSONDecoder = decoder
	return t
}

// ModelHTTPInfer Call Triton Infer with HTTP.
func (t *TritonClientService) ModelHTTPInfer(
	requestBody []byte,
	modelName, modelVersion string,
	decoderFunc DecoderFunc,
	params ...interface{},
) ([]interface{}, error) {
	// get infer response.
	modelInferResponse, inferErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForModelPrefix+modelName+TritonAPIForModelVersionPrefix+modelVersion+"/infer",
		requestBody)
	defer fasthttp.ReleaseResponse(modelInferResponse)

	if modelInferResponse == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, errors.New("modelInferResponse is nil"))
	}

	if inferErr != nil || modelInferResponse.StatusCode() != fasthttp.StatusOK {
		if inferErr == nil && modelInferResponse.Body() != nil {
			inferErr = errors.New("Triton error resp: " + string(modelInferResponse.Body()))
		}
		return nil, t.httpErrorHandler(modelInferResponse.StatusCode(), inferErr)
	}
	// decode Result.
	response, decodeErr := decoderFunc(modelInferResponse.Body(), params...)
	if decodeErr != nil {
		return nil, t.decodeFuncErrorHandler(decodeErr, false)
	}
	return response, nil
}

// ModelGRPCInfer Call Triton Infer with GRPC.
func (t *TritonClientService) ModelGRPCInfer(
	inferInputs []*ModelInferRequest_InferInputTensor,
	inferOutputs []*ModelInferRequest_InferRequestedOutputTensor,
	rawInputs [][]byte,
	modelName, modelVersion string,
	decoderFunc DecoderFunc,
	params ...interface{},
) ([]interface{}, error) {
	// Get infer response.
	modelInferResponse, inferErr := t.modelGRPCInfer(
		inferInputs, inferOutputs, rawInputs, modelName, modelVersion)
	if inferErr != nil {
		return nil, t.grpcErrorHandler(inferErr)
	}
	// decode Result.
	response, decodeErr := decoderFunc(modelInferResponse, params...)
	if decodeErr != nil {
		return nil, t.decodeFuncErrorHandler(decodeErr, true)
	}
	return response, nil
}

// CheckServerAlive check server is alive.
func (t *TritonClientService) CheckServerAlive() (bool, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// server alive.
		serverLiveResponse, serverAliveErr := t.grpcClient.ServerLive(ctx, &ServerLiveRequest{})
		if serverAliveErr != nil {
			return false, t.grpcErrorHandler(serverAliveErr)
		}
		return serverLiveResponse.Live, nil
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(t.getServerURL()+TritonAPIForServerIsLive, nil)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return false, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return false, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	return true, nil
}

// CheckServerReady check server is ready.
func (t *TritonClientService) CheckServerReady() (bool, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// server ready.
		serverReadyResponse, serverReadyErr := t.grpcClient.ServerReady(ctx, &ServerReadyRequest{})
		if serverReadyErr != nil {
			return false, t.grpcErrorHandler(serverReadyErr)
		}
		return serverReadyResponse.Ready, nil
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(t.getServerURL()+TritonAPIForServerIsReady, nil)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return false, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return false, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	return true, nil
}

// CheckModelReady check model is ready.
func (t *TritonClientService) CheckModelReady(modelName, modelVersion string) (bool, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// model ready
		modelReadyResponse, modelReadyErr := t.grpcClient.ModelReady(
			ctx, &ModelReadyRequest{Name: modelName, Version: modelVersion})
		if modelReadyErr != nil {
			return false, t.grpcErrorHandler(modelReadyErr)
		}
		return modelReadyResponse.Ready, nil
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForModelPrefix+modelName+TritonAPIForModelVersionPrefix+modelVersion+"/ready",
		nil)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return false, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return false, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	return true, nil
}

// ServerMetadata Get server metadata.
func (t *TritonClientService) ServerMetadata() (*ServerMetadataResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// server metadata
		serverMetadataResponse, serverMetaErr := t.grpcClient.ServerMetadata(ctx, &ServerMetadataRequest{})
		return serverMetadataResponse, t.grpcErrorHandler(serverMetaErr)
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(t.getServerURL()+TritonAPIPrefix, nil)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	serverMetadataResponse := new(ServerMetadataResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &serverMetadataResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return serverMetadataResponse, nil
}

// ModelMetadataRequest Get model metadata.
func (t *TritonClientService) ModelMetadataRequest(modelName, modelVersion string) (*ModelMetadataResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// model metadata
		modelMetadataResponse, modelMetaErr := t.grpcClient.ModelMetadata(
			ctx, &ModelMetadataRequest{Name: modelName, Version: modelVersion})
		return modelMetadataResponse, t.grpcErrorHandler(modelMetaErr)
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForModelPrefix+modelName+TritonAPIForModelVersionPrefix+modelVersion, nil)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	modelMetadataResponse := new(ModelMetadataResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &modelMetadataResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return modelMetadataResponse, nil
}

// ModelIndex Get model repo index.
func (t *TritonClientService) ModelIndex(repoName string, isReady bool) (*RepositoryIndexResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// The name of the repository. If empty the index is returned for all repositories.
		repositoryIndexResponse, modelIndexErr := t.grpcClient.RepositoryIndex(
			ctx, &RepositoryIndexRequest{RepositoryName: repoName, Ready: isReady})
		return repositoryIndexResponse, t.grpcErrorHandler(modelIndexErr)
	}
	reqBody, jsonEncodeErr := t.JsonMarshal(&ModelIndexRequestHTTPObj{repoName, isReady})
	if jsonEncodeErr != nil {
		return nil, jsonEncodeErr
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(t.getServerURL()+TritonAPIForRepoIndex, reqBody)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	repositoryIndexResponse := new(RepositoryIndexResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &repositoryIndexResponse.Models); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return repositoryIndexResponse, nil
}

// ModelConfiguration Get model configuration.
func (t *TritonClientService) ModelConfiguration(modelName, modelVersion string) (*ModelConfigResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		modelConfigResponse, getModelConfigErr := t.grpcClient.ModelConfig(
			ctx, &ModelConfigRequest{Name: modelName, Version: modelVersion})
		return modelConfigResponse, t.grpcErrorHandler(getModelConfigErr)
	}
	apiResp, httpErr := t.makeHTTPGetRequestWithDoTimeout(
		t.getServerURL() + TritonAPIForModelPrefix + modelName +
			TritonAPIForModelVersionPrefix + modelVersion + "/config")
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	modelConfigResponse := new(ModelConfigResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &modelConfigResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return modelConfigResponse, nil
}

// ModelInferStats Get Model infer stats.
func (t *TritonClientService) ModelInferStats(modelName, modelVersion string) (*ModelStatisticsResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		modelStatisticsResponse, getInferStatsErr := t.grpcClient.ModelStatistics(
			ctx, &ModelStatisticsRequest{Name: modelName, Version: modelVersion})
		return modelStatisticsResponse, t.grpcErrorHandler(getInferStatsErr)
	}
	apiResp, httpErr := t.makeHTTPGetRequestWithDoTimeout(
		t.getServerURL() + TritonAPIForModelPrefix + modelName + TritonAPIForModelVersionPrefix + modelVersion + "/stats")
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	modelStatisticsResponse := new(ModelStatisticsResponse)
	jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &modelStatisticsResponse)
	if jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return modelStatisticsResponse, nil
}

// ModelLoadWithHTTP Load Model with http
// modelConfigBody ==>
// https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md#examples
func (t *TritonClientService) ModelLoadWithHTTP(modelName string, modelConfigBody []byte) (*RepositoryModelLoadResponse, error) {
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForRepoModelPrefix+modelName+"/load", modelConfigBody)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	repositoryModelLoadResponse := new(RepositoryModelLoadResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &repositoryModelLoadResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return repositoryModelLoadResponse, nil
}

// ModelLoadWithGRPC Load Model with grpc.
func (t *TritonClientService) ModelLoadWithGRPC(repoName, modelName string, modelConfigBody map[string]*ModelRepositoryParameter) (*RepositoryModelLoadResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
	defer cancel()
	// The name of the repository to load from. If empty the model is loaded from any repository.
	loadResponse, loadErr := t.grpcClient.RepositoryModelLoad(ctx, &RepositoryModelLoadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return loadResponse, t.grpcErrorHandler(loadErr)
}

// ModelUnloadWithHTTP Unload model with http
// modelConfigBody if not is nil.
func (t *TritonClientService) ModelUnloadWithHTTP(modelName string, modelConfigBody []byte) (*RepositoryModelUnloadResponse, error) {
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForRepoModelPrefix+modelName+"/unload", modelConfigBody)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	repositoryModelUnloadResponse := new(RepositoryModelUnloadResponse)
	jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &repositoryModelUnloadResponse)
	if jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return repositoryModelUnloadResponse, nil
}

// ModelUnloadWithGRPC Unload model with grpc
// modelConfigBody if not is nil.
func (t *TritonClientService) ModelUnloadWithGRPC(repoName, modelName string, modelConfigBody map[string]*ModelRepositoryParameter) (*RepositoryModelUnloadResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
	defer cancel()

	unloadResponse, unloadErr := t.grpcClient.RepositoryModelUnload(ctx, &RepositoryModelUnloadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return unloadResponse, t.grpcErrorHandler(unloadErr)
}

// ShareMemoryStatus Get share memory / cuda memory status.
// Response: CudaSharedMemoryStatusResponse / SystemSharedMemoryStatusResponse.
func (t *TritonClientService) ShareMemoryStatus(isCUDA bool, regionName string) (interface{}, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		if isCUDA {
			// CUDA Memory
			cudaSharedMemoryStatusResponse, cudaStatusErr := t.grpcClient.CudaSharedMemoryStatus(
				ctx, &CudaSharedMemoryStatusRequest{Name: regionName})

			return cudaSharedMemoryStatusResponse, t.grpcErrorHandler(cudaStatusErr)
		}
		// System Memory
		systemSharedMemoryStatusResponse, systemStatusErr := t.grpcClient.SystemSharedMemoryStatus(
			ctx, &SystemSharedMemoryStatusRequest{Name: regionName})
		if systemStatusErr != nil {
			return nil, t.grpcErrorHandler(systemStatusErr)
		}
		return systemSharedMemoryStatusResponse, nil
	}
	// SetRequestURI
	var uri string
	if isCUDA {
		uri = t.getServerURL() + TritonAPIForCudaMemoryRegionPrefix + regionName + "/status"
	} else {
		uri = t.getServerURL() + TritonAPIForSystemMemoryRegionPrefix + regionName + "/status"
	}
	apiResp, httpErr := t.makeHTTPGetRequestWithDoTimeout(uri)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return false, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	// Parse Response
	if isCUDA {
		cudaSharedMemoryStatusResponse := new(CudaSharedMemoryStatusResponse)
		if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &cudaSharedMemoryStatusResponse); jsonDecodeErr != nil {
			return nil, jsonDecodeErr
		}
		return cudaSharedMemoryStatusResponse, nil
	}
	systemSharedMemoryStatusResponse := new(SystemSharedMemoryStatusResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &systemSharedMemoryStatusResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return systemSharedMemoryStatusResponse, nil
}

// ShareCUDAMemoryRegister cuda share memory register.
func (t *TritonClientService) ShareCUDAMemoryRegister(regionName string, cudaRawHandle []byte, cudaDeviceID int64, byteSize uint64) (*CudaSharedMemoryRegisterResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// CUDA Memory
		cudaSharedMemoryRegisterResponse, registerErr := t.grpcClient.CudaSharedMemoryRegister(
			ctx, &CudaSharedMemoryRegisterRequest{
				Name:      regionName,
				RawHandle: cudaRawHandle,
				DeviceId:  cudaDeviceID,
				ByteSize:  byteSize,
			},
		)
		return cudaSharedMemoryRegisterResponse, t.grpcErrorHandler(registerErr)
	}
	reqBody, jsonEncodeErr := t.JsonMarshal(
		&CudaMemoryRegisterBodyHTTPObj{cudaRawHandle, cudaDeviceID, byteSize})
	if jsonEncodeErr != nil {
		return nil, jsonEncodeErr
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForCudaMemoryRegionPrefix+regionName+"/register", reqBody)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	cudaSharedMemoryRegisterResponse := new(CudaSharedMemoryRegisterResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &cudaSharedMemoryRegisterResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return cudaSharedMemoryRegisterResponse, nil
}

// ShareCUDAMemoryUnRegister cuda share memory unregister.
func (t *TritonClientService) ShareCUDAMemoryUnRegister(regionName string) (*CudaSharedMemoryUnregisterResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// CUDA Memory
		cudaSharedMemoryUnRegisterResponse, unRegisterErr := t.grpcClient.CudaSharedMemoryUnregister(
			ctx, &CudaSharedMemoryUnregisterRequest{Name: regionName})
		return cudaSharedMemoryUnRegisterResponse, t.grpcErrorHandler(unRegisterErr)
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForCudaMemoryRegionPrefix+regionName+"/unregister", nil)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	cudaSharedMemoryUnregisterResponse := new(CudaSharedMemoryUnregisterResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &cudaSharedMemoryUnregisterResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return cudaSharedMemoryUnregisterResponse, nil
}

// ShareSystemMemoryRegister system share memory register.
func (t *TritonClientService) ShareSystemMemoryRegister(regionName, cpuMemRegionKey string, byteSize, cpuMemOffset uint64) (*SystemSharedMemoryRegisterResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// System Memory
		systemSharedMemoryRegisterResponse, registerErr := t.grpcClient.SystemSharedMemoryRegister(
			ctx, &SystemSharedMemoryRegisterRequest{
				Name:     regionName,
				Key:      cpuMemRegionKey,
				Offset:   cpuMemOffset,
				ByteSize: byteSize,
			},
		)
		return systemSharedMemoryRegisterResponse, t.grpcErrorHandler(registerErr)
	}
	reqBody, jsonEncodeErr := t.JsonMarshal(
		&SystemMemoryRegisterBodyHTTPObj{cpuMemRegionKey, cpuMemOffset, byteSize})
	if jsonEncodeErr != nil {
		return nil, jsonEncodeErr
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForSystemMemoryRegionPrefix+regionName+"/register", reqBody)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	systemSharedMemoryRegisterResponse := new(SystemSharedMemoryRegisterResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &systemSharedMemoryRegisterResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return systemSharedMemoryRegisterResponse, nil
}

// ShareSystemMemoryUnRegister system share memory unregister.
func (t *TritonClientService) ShareSystemMemoryUnRegister(regionName string) (*SystemSharedMemoryUnregisterResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// System Memory
		systemSharedMemoryUnRegisterResponse, unRegisterErr := t.grpcClient.SystemSharedMemoryUnregister(
			ctx, &SystemSharedMemoryUnregisterRequest{Name: regionName})
		return systemSharedMemoryUnRegisterResponse, t.grpcErrorHandler(unRegisterErr)
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForSystemMemoryRegionPrefix+regionName+"/unregister", nil)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	systemSharedMemoryUnregisterResponse := new(SystemSharedMemoryUnregisterResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), &systemSharedMemoryUnregisterResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return systemSharedMemoryUnregisterResponse, nil
}

// GetModelTracingSetting get model tracing setting.
func (t *TritonClientService) GetModelTracingSetting(modelName string) (*TraceSettingResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		// Tracing
		traceSettingResponse, getTraceSettingErr := t.grpcClient.TraceSetting(
			ctx, &TraceSettingRequest{ModelName: modelName})
		return traceSettingResponse, t.grpcErrorHandler(getTraceSettingErr)
	}
	apiResp, httpErr := t.makeHTTPGetRequestWithDoTimeout(
		t.getServerURL() + TritonAPIForModelPrefix + modelName + "/trace/setting")
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	traceSettingResponse := new(TraceSettingResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), traceSettingResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return traceSettingResponse, nil
}

// SetModelTracingSetting set model tracing setting.
// Param: settingMap ==>
// https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_trace.md#trace-setting-response-json-object
func (t *TritonClientService) SetModelTracingSetting(
	modelName string, settingMap map[string]*TraceSettingRequest_SettingValue,
) (*TraceSettingResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), t.apiTimeout)
		defer cancel()

		traceSettingResponse, setTraceSettingErr := t.grpcClient.TraceSetting(
			ctx, &TraceSettingRequest{ModelName: modelName, Settings: settingMap})
		return traceSettingResponse, t.grpcErrorHandler(setTraceSettingErr)
	}
	// Experimental
	reqBody, jsonEncodeErr := t.JsonMarshal(&TraceSettingRequestHTTPObj{settingMap})
	if jsonEncodeErr != nil {
		return nil, jsonEncodeErr
	}
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(t.getServerURL()+TritonAPIForModelPrefix+modelName+"/trace/setting", reqBody)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return nil, t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return nil, t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	traceSettingResponse := new(TraceSettingResponse)
	if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), traceSettingResponse); jsonDecodeErr != nil {
		return nil, jsonDecodeErr
	}
	return traceSettingResponse, nil
}

// SetSecondaryServerURL Set secondary server url
func (t *TritonClientService) SetSecondaryServerURL(url string) {
	t.secondaryServerURL = url
}

// SetAPIRequestTimeout Set API request timeout.
func (t *TritonClientService) SetAPIRequestTimeout(timeout time.Duration) {
	t.apiTimeout = timeout
}

// ShutdownTritonConnection shutdown http and grpc connection.
func (t *TritonClientService) ShutdownTritonConnection() (disconnectionErr error) {
	if t.grpcConn != nil {
		disconnectionErr = t.disconnectToTritonWithGRPC()
	}
	if t.httpClient != nil {
		t.disconnectToTritonWithHTTP()
	}
	if disconnectionErr != nil {
		return errors.New("[Triton]DisconnectionError: " + disconnectionErr.Error())
	}
	return nil
}

// NewTritonClientWithOnlyHTTP init triton client.
func NewTritonClientWithOnlyHTTP(uri string, httpClient *fasthttp.Client) *TritonClientService {
	client := &TritonClientService{serverURL: uri, JSONEncoder: json.Marshal, JSONDecoder: json.Unmarshal}
	client.setHTTPConnection(httpClient)
	return client
}

// NewTritonClientWithOnlyGRPC init triton client.
func NewTritonClientWithOnlyGRPC(grpcConn *grpc.ClientConn) *TritonClientService {
	if grpcConn == nil {
		return nil
	}
	client := &TritonClientService{
		grpcConn:    grpcConn,
		grpcClient:  NewGRPCInferenceServiceClient(grpcConn),
		apiTimeout:  DefaultHTTPClientReadTimeout,
		JSONEncoder: json.Marshal,
		JSONDecoder: json.Unmarshal,
	}
	return client
}

// NewTritonClientForAll init triton client with http and grpc.
func NewTritonClientForAll(httpServerURL string, httpClient *fasthttp.Client, grpcConn *grpc.ClientConn) *TritonClientService {
	client := &TritonClientService{
		serverURL:   httpServerURL,
		grpcConn:    grpcConn,
		grpcClient:  NewGRPCInferenceServiceClient(grpcConn),
		apiTimeout:  DefaultHTTPClientReadTimeout,
		JSONEncoder: json.Marshal,
		JSONDecoder: json.Unmarshal,
	}
	client.setHTTPConnection(httpClient)

	return client
}
