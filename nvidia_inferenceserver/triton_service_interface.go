package nvidia_inferenceserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
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

// pathEscape URL-encodes a path segment to prevent path injection.
func pathEscape(s string) string {
	return url.PathEscape(s)
}

// ensureCtx wraps ctx with a timeout if it doesn't already have a deadline.
// If ctx is nil or context.Background(), it returns a new context with the service's apiTimeout.
func (t *TritonClientService) ensureCtx(ctx context.Context) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, t.apiTimeout)
}

// DecoderFunc Infer Callback Function.
type DecoderFunc func(response any, params ...any) ([]any, error)

// TritonService Service interface.
type TritonService interface {
	// JsonMarshal Json Encoder
	JsonMarshal(v any) ([]byte, error)
	// JsonUnmarshal Json Decoder
	JsonUnmarshal(data []byte, v any) error

	// CheckServerAlive Check triton inference server is alive.
	CheckServerAlive(ctx context.Context) (bool, error)
	// CheckServerReady Check triton inference server is ready.
	CheckServerReady(ctx context.Context) (bool, error)
	// CheckModelReady Check triton inference server's model is ready.
	CheckModelReady(ctx context.Context, modelName, modelVersion string) (bool, error)
	// ServerMetadata Get triton inference server metadata.
	ServerMetadata(ctx context.Context) (*ServerMetadataResponse, error)
	// ModelGRPCInfer Call triton inference server infer with GRPC.
	ModelGRPCInfer(
		ctx context.Context,
		inferInputs []*ModelInferRequest_InferInputTensor,
		inferOutputs []*ModelInferRequest_InferRequestedOutputTensor,
		rawInputs [][]byte,
		modelName, modelVersion string,
		decoderFunc DecoderFunc,
		params ...any,
	) ([]any, error)
	// ModelHTTPInfer Call triton inference server infer with HTTP.
	ModelHTTPInfer(
		ctx context.Context,
		requestBody []byte,
		modelName, modelVersion string,
		decoderFunc DecoderFunc,
		params ...any,
	) ([]any, error)
	// ModelMetadataRequest Get triton inference server's model metadata.
	ModelMetadataRequest(ctx context.Context, modelName, modelVersion string) (*ModelMetadataResponse, error)
	// ModelIndex Get triton inference server model index.
	ModelIndex(ctx context.Context, repoName string, isReady bool) (*RepositoryIndexResponse, error)
	// ModelConfiguration Get triton inference server model configuration.
	ModelConfiguration(ctx context.Context, modelName, modelVersion string) (*ModelConfigResponse, error)
	// ModelInferStats Get triton inference server model infer stats.
	ModelInferStats(ctx context.Context, modelName, modelVersion string) (*ModelStatisticsResponse, error)
	// ModelLoadWithHTTP Load model with HTTP.
	ModelLoadWithHTTP(ctx context.Context, modelName string, modelConfigBody []byte) (*RepositoryModelLoadResponse, error)
	// ModelLoadWithGRPC Load model with gRPC.
	ModelLoadWithGRPC(ctx context.Context, repoName, modelName string, modelConfigBody map[string]*ModelRepositoryParameter) (*RepositoryModelLoadResponse, error)
	// ModelUnloadWithHTTP Unload model with HTTP.
	ModelUnloadWithHTTP(ctx context.Context, modelName string, modelConfigBody []byte) (*RepositoryModelUnloadResponse, error)
	// ModelUnloadWithGRPC Unload model with gRPC.
	ModelUnloadWithGRPC(ctx context.Context, repoName, modelName string, modelConfigBody map[string]*ModelRepositoryParameter) (*RepositoryModelUnloadResponse, error)
	// ShareCUDAMemoryStatus Get CUDA shared memory status.
	ShareCUDAMemoryStatus(ctx context.Context, regionName string) (*CudaSharedMemoryStatusResponse, error)
	// ShareSystemMemoryStatus Get system shared memory status.
	ShareSystemMemoryStatus(ctx context.Context, regionName string) (*SystemSharedMemoryStatusResponse, error)
	// ShareCUDAMemoryRegister Register share cuda memory.
	ShareCUDAMemoryRegister(ctx context.Context, regionName string, cudaRawHandle []byte, cudaDeviceID int64, byteSize uint64) (*CudaSharedMemoryRegisterResponse, error)
	// ShareCUDAMemoryUnRegister Unregister share cuda memory
	ShareCUDAMemoryUnRegister(ctx context.Context, regionName string) (*CudaSharedMemoryUnregisterResponse, error)
	// ShareSystemMemoryRegister Register system share memory.
	ShareSystemMemoryRegister(ctx context.Context, regionName, cpuMemRegionKey string, byteSize, cpuMemOffset uint64) (*SystemSharedMemoryRegisterResponse, error)
	// ShareSystemMemoryUnRegister Unregister system share memory.
	ShareSystemMemoryUnRegister(ctx context.Context, regionName string) (*SystemSharedMemoryUnregisterResponse, error)
	// GetModelTracingSetting get the current trace setting.
	GetModelTracingSetting(ctx context.Context, modelName string) (*TraceSettingResponse, error)
	// SetModelTracingSetting set the current trace setting.
	SetModelTracingSetting(ctx context.Context, modelName string, settingMap map[string]*TraceSettingRequest_SettingValue) (*TraceSettingResponse, error)

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
		t.httpClient = &fasthttp.Client{
			MaxConnsPerHost: DefaultHTTPClientMaxConnPerHost,
			ReadTimeout:     DefaultHTTPClientReadTimeout,
			WriteTimeout:    DefaultHTTPClientWriteTimeout,
		}
		return
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
	ctx context.Context,
	inferInputs []*ModelInferRequest_InferInputTensor,
	inferOutputs []*ModelInferRequest_InferRequestedOutputTensor,
	rawInputs [][]byte,
	modelName, modelVersion string,
) (*ModelInferResponse, error) {
	if t.grpcClient == nil {
		return nil, errors.New("[GRPC]grpc client is nil")
	}
	ctx, cancel := t.ensureCtx(ctx)
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
		return nil, fmt.Errorf("[GRPC]inferErr: %w", inferErr)
	}
	return modelInferResponse, nil
}

// httpErrorHandler HTTP Error Handler.
func (t *TritonClientService) httpErrorHandler(statusCode int, httpErr error) error {
	if httpErr != nil {
		return fmt.Errorf("[HTTP]code: %d; error: %w", statusCode, httpErr)
	}
	if statusCode != http.StatusOK {
		return fmt.Errorf("[HTTP]unexpected status code: %d", statusCode)
	}
	return nil
}

// grpcErrorHandler GRPC Error Handler.
func (t *TritonClientService) grpcErrorHandler(grpcErr error) error {
	if grpcErr != nil {
		return fmt.Errorf("[GRPC]error: %w", grpcErr)
	}
	return nil
}

// decodeFuncErrorHandler DecodeFunc Error Handler.
func (t *TritonClientService) decodeFuncErrorHandler(err error, isGRPC bool) error {
	if isGRPC {
		return fmt.Errorf("[GRPC]decodeFunc error: %w", err)
	}
	return fmt.Errorf("[HTTP]decodeFunc error: %w", err)
}

// httpGetAndDecode performs an HTTP GET request, checks the response, and decodes the JSON body into result.
func (t *TritonClientService) httpGetAndDecode(url string, result any) error {
	apiResp, httpErr := t.makeHTTPGetRequestWithDoTimeout(url)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	if result != nil {
		if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), result); jsonDecodeErr != nil {
			return jsonDecodeErr
		}
	}
	return nil
}

// httpPostAndDecode performs an HTTP POST request, checks the response, and decodes the JSON body into result.
func (t *TritonClientService) httpPostAndDecode(url string, reqBody []byte, result any) error {
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(url, reqBody)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return t.httpErrorHandler(http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return t.httpErrorHandler(apiResp.StatusCode(), httpErr)
	}
	if result != nil {
		if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), result); jsonDecodeErr != nil {
			return jsonDecodeErr
		}
	}
	return nil
}

///////////////////////////////////////////// expose API below /////////////////////////////////////////////

// JsonMarshal Json Encoder
func (t *TritonClientService) JsonMarshal(v any) ([]byte, error) {
	return t.JSONEncoder(v)
}

// JsonUnmarshal Json Decoder
func (t *TritonClientService) JsonUnmarshal(data []byte, v any) error {
	return t.JSONDecoder(data, v)
}

// SetJSONEncoder set json encoder
func (t *TritonClientService) SetJSONEncoder(encoder utils.JSONMarshal) *TritonClientService {
	t.JSONEncoder = encoder
	return t
}

// SetJSONDecoder set json decoder
func (t *TritonClientService) SetJSONDecoder(decoder utils.JSONUnmarshal) *TritonClientService {
	t.JSONDecoder = decoder
	return t
}

// ModelHTTPInfer Call Triton Infer with HTTP.
func (t *TritonClientService) ModelHTTPInfer(
	ctx context.Context,
	requestBody []byte,
	modelName, modelVersion string,
	decoderFunc DecoderFunc,
	params ...any,
) ([]any, error) {
	// Check context cancellation
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("[HTTP]context error: %w", err)
	}
	// get infer response.
	modelInferResponse, inferErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+TritonAPIForModelVersionPrefix+pathEscape(modelVersion)+"/infer",
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
	ctx context.Context,
	inferInputs []*ModelInferRequest_InferInputTensor,
	inferOutputs []*ModelInferRequest_InferRequestedOutputTensor,
	rawInputs [][]byte,
	modelName, modelVersion string,
	decoderFunc DecoderFunc,
	params ...any,
) ([]any, error) {
	// Get infer response.
	modelInferResponse, inferErr := t.modelGRPCInfer(
		ctx, inferInputs, inferOutputs, rawInputs, modelName, modelVersion)
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
func (t *TritonClientService) CheckServerAlive(ctx context.Context) (bool, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
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
func (t *TritonClientService) CheckServerReady(ctx context.Context) (bool, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
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
func (t *TritonClientService) CheckModelReady(ctx context.Context, modelName, modelVersion string) (bool, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
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
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+TritonAPIForModelVersionPrefix+pathEscape(modelVersion)+"/ready",
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
func (t *TritonClientService) ServerMetadata(ctx context.Context) (*ServerMetadataResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		// server metadata
		serverMetadataResponse, serverMetaErr := t.grpcClient.ServerMetadata(ctx, &ServerMetadataRequest{})
		return serverMetadataResponse, t.grpcErrorHandler(serverMetaErr)
	}
	serverMetadataResponse := new(ServerMetadataResponse)
	return serverMetadataResponse, t.httpPostAndDecode(t.getServerURL()+TritonAPIPrefix, nil, serverMetadataResponse)
}

// ModelMetadataRequest Get model metadata.
func (t *TritonClientService) ModelMetadataRequest(ctx context.Context, modelName, modelVersion string) (*ModelMetadataResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		// model metadata
		modelMetadataResponse, modelMetaErr := t.grpcClient.ModelMetadata(
			ctx, &ModelMetadataRequest{Name: modelName, Version: modelVersion})
		return modelMetadataResponse, t.grpcErrorHandler(modelMetaErr)
	}
	modelMetadataResponse := new(ModelMetadataResponse)
	return modelMetadataResponse, t.httpPostAndDecode(
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+TritonAPIForModelVersionPrefix+pathEscape(modelVersion),
		nil, modelMetadataResponse)
}

// ModelIndex Get model repo index.
func (t *TritonClientService) ModelIndex(ctx context.Context, repoName string, isReady bool) (*RepositoryIndexResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		repositoryIndexResponse, modelIndexErr := t.grpcClient.RepositoryIndex(
			ctx, &RepositoryIndexRequest{RepositoryName: repoName, Ready: isReady})
		return repositoryIndexResponse, t.grpcErrorHandler(modelIndexErr)
	}
	reqBody, jsonEncodeErr := t.JsonMarshal(&ModelIndexRequestHTTPObj{repoName, isReady})
	if jsonEncodeErr != nil {
		return nil, jsonEncodeErr
	}
	repositoryIndexResponse := new(RepositoryIndexResponse)
	if httpErr := t.httpPostAndDecode(t.getServerURL()+TritonAPIForRepoIndex, reqBody, &repositoryIndexResponse.Models); httpErr != nil {
		return nil, httpErr
	}
	return repositoryIndexResponse, nil
}

// ModelConfiguration Get model configuration.
func (t *TritonClientService) ModelConfiguration(ctx context.Context, modelName, modelVersion string) (*ModelConfigResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		modelConfigResponse, getModelConfigErr := t.grpcClient.ModelConfig(
			ctx, &ModelConfigRequest{Name: modelName, Version: modelVersion})
		return modelConfigResponse, t.grpcErrorHandler(getModelConfigErr)
	}
	modelConfigResponse := new(ModelConfigResponse)
	return modelConfigResponse, t.httpGetAndDecode(
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+
			TritonAPIForModelVersionPrefix+pathEscape(modelVersion)+"/config", modelConfigResponse)
}

// ModelInferStats Get Model infer stats.
func (t *TritonClientService) ModelInferStats(ctx context.Context, modelName, modelVersion string) (*ModelStatisticsResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		modelStatisticsResponse, getInferStatsErr := t.grpcClient.ModelStatistics(
			ctx, &ModelStatisticsRequest{Name: modelName, Version: modelVersion})
		return modelStatisticsResponse, t.grpcErrorHandler(getInferStatsErr)
	}
	modelStatisticsResponse := new(ModelStatisticsResponse)
	return modelStatisticsResponse, t.httpGetAndDecode(
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+
			TritonAPIForModelVersionPrefix+pathEscape(modelVersion)+"/stats", modelStatisticsResponse)
}

// ModelLoadWithHTTP Load Model with http
// modelConfigBody ==>
// https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md#examples
func (t *TritonClientService) ModelLoadWithHTTP(ctx context.Context, modelName string, modelConfigBody []byte) (*RepositoryModelLoadResponse, error) {
	repositoryModelLoadResponse := new(RepositoryModelLoadResponse)
	return repositoryModelLoadResponse, t.httpPostAndDecode(
		t.getServerURL()+TritonAPIForRepoModelPrefix+pathEscape(modelName)+"/load", modelConfigBody, repositoryModelLoadResponse)
}

// ModelLoadWithGRPC Load model with gRPC.
func (t *TritonClientService) ModelLoadWithGRPC(ctx context.Context, repoName, modelName string, modelConfigBody map[string]*ModelRepositoryParameter) (*RepositoryModelLoadResponse, error) {
	if t.grpcClient == nil {
		return nil, errors.New("[GRPC]grpc client is nil")
	}
	ctx, cancel := t.ensureCtx(ctx)
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
func (t *TritonClientService) ModelUnloadWithHTTP(ctx context.Context, modelName string, modelConfigBody []byte) (*RepositoryModelUnloadResponse, error) {
	repositoryModelUnloadResponse := new(RepositoryModelUnloadResponse)
	return repositoryModelUnloadResponse, t.httpPostAndDecode(
		t.getServerURL()+TritonAPIForRepoModelPrefix+pathEscape(modelName)+"/unload", modelConfigBody, repositoryModelUnloadResponse)
}

// ModelUnloadWithGRPC Unload model with gRPC.
func (t *TritonClientService) ModelUnloadWithGRPC(ctx context.Context, repoName, modelName string, modelConfigBody map[string]*ModelRepositoryParameter) (*RepositoryModelUnloadResponse, error) {
	if t.grpcClient == nil {
		return nil, errors.New("[GRPC]grpc client is nil")
	}
	ctx, cancel := t.ensureCtx(ctx)
	defer cancel()

	unloadResponse, unloadErr := t.grpcClient.RepositoryModelUnload(ctx, &RepositoryModelUnloadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return unloadResponse, t.grpcErrorHandler(unloadErr)
}

// ShareCUDAMemoryStatus Get CUDA shared memory status.
func (t *TritonClientService) ShareCUDAMemoryStatus(ctx context.Context, regionName string) (*CudaSharedMemoryStatusResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		cudaSharedMemoryStatusResponse, cudaStatusErr := t.grpcClient.CudaSharedMemoryStatus(
			ctx, &CudaSharedMemoryStatusRequest{Name: regionName})
		return cudaSharedMemoryStatusResponse, t.grpcErrorHandler(cudaStatusErr)
	}
	cudaSharedMemoryStatusResponse := new(CudaSharedMemoryStatusResponse)
	return cudaSharedMemoryStatusResponse, t.httpGetAndDecode(
		t.getServerURL()+TritonAPIForCudaMemoryRegionPrefix+pathEscape(regionName)+"/status", cudaSharedMemoryStatusResponse)
}

// ShareSystemMemoryStatus Get system shared memory status.
func (t *TritonClientService) ShareSystemMemoryStatus(ctx context.Context, regionName string) (*SystemSharedMemoryStatusResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		systemSharedMemoryStatusResponse, systemStatusErr := t.grpcClient.SystemSharedMemoryStatus(
			ctx, &SystemSharedMemoryStatusRequest{Name: regionName})
		return systemSharedMemoryStatusResponse, t.grpcErrorHandler(systemStatusErr)
	}
	systemSharedMemoryStatusResponse := new(SystemSharedMemoryStatusResponse)
	return systemSharedMemoryStatusResponse, t.httpGetAndDecode(
		t.getServerURL()+TritonAPIForSystemMemoryRegionPrefix+pathEscape(regionName)+"/status", systemSharedMemoryStatusResponse)
}

// ShareCUDAMemoryRegister cuda share memory register.
func (t *TritonClientService) ShareCUDAMemoryRegister(ctx context.Context, regionName string, cudaRawHandle []byte, cudaDeviceID int64, byteSize uint64) (*CudaSharedMemoryRegisterResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

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
	cudaSharedMemoryRegisterResponse := new(CudaSharedMemoryRegisterResponse)
	return cudaSharedMemoryRegisterResponse, t.httpPostAndDecode(
		t.getServerURL()+TritonAPIForCudaMemoryRegionPrefix+pathEscape(regionName)+"/register", reqBody, cudaSharedMemoryRegisterResponse)
}

// ShareCUDAMemoryUnRegister cuda share memory unregister.
func (t *TritonClientService) ShareCUDAMemoryUnRegister(ctx context.Context, regionName string) (*CudaSharedMemoryUnregisterResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		cudaSharedMemoryUnRegisterResponse, unRegisterErr := t.grpcClient.CudaSharedMemoryUnregister(
			ctx, &CudaSharedMemoryUnregisterRequest{Name: regionName})
		return cudaSharedMemoryUnRegisterResponse, t.grpcErrorHandler(unRegisterErr)
	}
	cudaSharedMemoryUnregisterResponse := new(CudaSharedMemoryUnregisterResponse)
	return cudaSharedMemoryUnregisterResponse, t.httpPostAndDecode(
		t.getServerURL()+TritonAPIForCudaMemoryRegionPrefix+pathEscape(regionName)+"/unregister", nil, cudaSharedMemoryUnregisterResponse)
}

// ShareSystemMemoryRegister system share memory register.
func (t *TritonClientService) ShareSystemMemoryRegister(ctx context.Context, regionName, cpuMemRegionKey string, byteSize, cpuMemOffset uint64) (*SystemSharedMemoryRegisterResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

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
	systemSharedMemoryRegisterResponse := new(SystemSharedMemoryRegisterResponse)
	return systemSharedMemoryRegisterResponse, t.httpPostAndDecode(
		t.getServerURL()+TritonAPIForSystemMemoryRegionPrefix+pathEscape(regionName)+"/register", reqBody, systemSharedMemoryRegisterResponse)
}

// ShareSystemMemoryUnRegister system share memory unregister.
func (t *TritonClientService) ShareSystemMemoryUnRegister(ctx context.Context, regionName string) (*SystemSharedMemoryUnregisterResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		systemSharedMemoryUnRegisterResponse, unRegisterErr := t.grpcClient.SystemSharedMemoryUnregister(
			ctx, &SystemSharedMemoryUnregisterRequest{Name: regionName})
		return systemSharedMemoryUnRegisterResponse, t.grpcErrorHandler(unRegisterErr)
	}
	systemSharedMemoryUnregisterResponse := new(SystemSharedMemoryUnregisterResponse)
	return systemSharedMemoryUnregisterResponse, t.httpPostAndDecode(
		t.getServerURL()+TritonAPIForSystemMemoryRegionPrefix+pathEscape(regionName)+"/unregister", nil, systemSharedMemoryUnregisterResponse)
}

// GetModelTracingSetting get model tracing setting.
func (t *TritonClientService) GetModelTracingSetting(ctx context.Context, modelName string) (*TraceSettingResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		traceSettingResponse, getTraceSettingErr := t.grpcClient.TraceSetting(
			ctx, &TraceSettingRequest{ModelName: modelName})
		return traceSettingResponse, t.grpcErrorHandler(getTraceSettingErr)
	}
	traceSettingResponse := new(TraceSettingResponse)
	return traceSettingResponse, t.httpGetAndDecode(
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+"/trace/setting", traceSettingResponse)
}

// SetModelTracingSetting set model tracing setting.
// Param: settingMap ==>
// https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_trace.md#trace-setting-response-json-object
func (t *TritonClientService) SetModelTracingSetting(
	ctx context.Context, modelName string, settingMap map[string]*TraceSettingRequest_SettingValue,
) (*TraceSettingResponse, error) {
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		traceSettingResponse, setTraceSettingErr := t.grpcClient.TraceSetting(
			ctx, &TraceSettingRequest{ModelName: modelName, Settings: settingMap})
		return traceSettingResponse, t.grpcErrorHandler(setTraceSettingErr)
	}
	reqBody, jsonEncodeErr := t.JsonMarshal(&TraceSettingRequestHTTPObj{settingMap})
	if jsonEncodeErr != nil {
		return nil, jsonEncodeErr
	}
	traceSettingResponse := new(TraceSettingResponse)
	return traceSettingResponse, t.httpPostAndDecode(
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+"/trace/setting", reqBody, traceSettingResponse)
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
	var errs []error
	if t.grpcConn != nil {
		if err := t.disconnectToTritonWithGRPC(); err != nil {
			errs = append(errs, err)
		}
	}
	if t.httpClient != nil {
		t.disconnectToTritonWithHTTP()
	}
	return errors.Join(errs...)
}

// NewTritonClientWithOnlyHTTP init triton client.
func NewTritonClientWithOnlyHTTP(uri string, httpClient *fasthttp.Client) *TritonClientService {
	client := &TritonClientService{
		serverURL:   uri,
		apiTimeout:  DefaultHTTPClientReadTimeout,
		JSONEncoder: json.Marshal,
		JSONDecoder: json.Unmarshal,
	}
	client.setHTTPConnection(httpClient)
	return client
}

// NewTritonClientWithOnlyGRPC init triton client.
func NewTritonClientWithOnlyGRPC(grpcConn *grpc.ClientConn) (*TritonClientService, error) {
	if grpcConn == nil {
		return nil, errors.New("[GRPC]grpc connection is nil")
	}
	client := &TritonClientService{
		grpcConn:    grpcConn,
		grpcClient:  NewGRPCInferenceServiceClient(grpcConn),
		apiTimeout:  DefaultHTTPClientReadTimeout,
		JSONEncoder: json.Marshal,
		JSONDecoder: json.Unmarshal,
	}
	return client, nil
}

// NewTritonClientForAll init triton client with http and grpc.
func NewTritonClientForAll(httpServerURL string, httpClient *fasthttp.Client, grpcConn *grpc.ClientConn) *TritonClientService {
	client := &TritonClientService{
		serverURL:   httpServerURL,
		apiTimeout:  DefaultHTTPClientReadTimeout,
		JSONEncoder: json.Marshal,
		JSONDecoder: json.Unmarshal,
	}
	if grpcConn != nil {
		client.grpcConn = grpcConn
		client.grpcClient = NewGRPCInferenceServiceClient(grpcConn)
	}
	client.setHTTPConnection(httpClient)

	return client
}
