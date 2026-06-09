package nvidia_inferenceserver

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
)

// Ensure TritonClientService implements TritonService at compile time.
var _ TritonService = (*TritonClientService)(nil)

const (
	DefaultHTTPClientReadTimeout                = 5 * time.Second
	DefaultHTTPClientWriteTimeout               = 5 * time.Second
	DefaultHTTPClientMaxConnPerHost      int    = 16384
	HTTPPrefix                           string = "http://"
	HTTPSPrefix                          string = "https://"
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
	return context.WithTimeout(ctx, t.getAPITimeout())
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
	mu sync.RWMutex

	serverURL          string
	secondaryServerURL atomic.Value // stores string
	apiTimeoutNanos    atomic.Int64 // stores time.Duration as nanoseconds

	grpcConn   *grpc.ClientConn
	grpcClient GRPCInferenceServiceClient
	httpClient *fasthttp.Client

	// Default: json.Marshal
	JSONEncoder utils.JSONMarshal
	// Default: json.Unmarshal
	JSONDecoder utils.JSONUnmarshal

	// Logger for observability. Default: slog.Default() (no-op if not configured).
	logger *slog.Logger
}

// ClientOption configures a TritonClientService.
type ClientOption func(*TritonClientService)

// WithLogger sets a structured logger for the client.
// If not set, a default no-op logger is used.
func WithLogger(logger *slog.Logger) ClientOption {
	return func(c *TritonClientService) {
		c.logger = logger
	}
}

// WithTimeout sets the default API request timeout.
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *TritonClientService) {
		c.apiTimeoutNanos.Store(int64(timeout))
	}
}

// WithJSONEncoder sets a custom JSON encoder (e.g., json-iterator, sonic).
func WithJSONEncoder(encoder utils.JSONMarshal) ClientOption {
	return func(c *TritonClientService) {
		c.JSONEncoder = encoder
	}
}

// WithJSONDecoder sets a custom JSON decoder (e.g., json-iterator, sonic).
func WithJSONDecoder(decoder utils.JSONUnmarshal) ClientOption {
	return func(c *TritonClientService) {
		c.JSONDecoder = decoder
	}
}

// WithHTTPClient sets a custom fasthttp.Client.
// This is useful for configuring TLS, custom timeouts, or connection pooling.
//
// Example with TLS:
//
//	tlsConfig := &tls.Config{RootCAs: certPool}
//	httpClient := &fasthttp.Client{TLSConfig: tlsConfig}
//	client := NewTritonClientWithOnlyHTTP("https://triton:8000", httpClient, WithLogger(logger))
func WithHTTPClient(client *fasthttp.Client) ClientOption {
	return func(c *TritonClientService) {
		c.httpClient = client
	}
}

// applyOptions applies functional options to a TritonClientService.
func (t *TritonClientService) applyOptions(opts []ClientOption) {
	for _, opt := range opts {
		opt(t)
	}
}

// Logger returns the client's logger.
func (t *TritonClientService) Logger() *slog.Logger {
	return t.logger
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
// Automatically detects if the URL already contains a scheme prefix.
// Lock-free: uses atomic.Value for secondaryServerURL.
func (t *TritonClientService) getServerURL() string {
	baseURL := t.serverURL
	if secondary, _ := t.secondaryServerURL.Load().(string); secondary != "" {
		baseURL = secondary
	}
	return ensureScheme(baseURL)
}

// ensureScheme adds http:// prefix if the URL doesn't already have a scheme.
func ensureScheme(rawURL string) string {
	if strings.HasPrefix(rawURL, "http://") || strings.HasPrefix(rawURL, "https://") {
		return rawURL
	}
	return HTTPPrefix + rawURL
}

// getAPITimeout returns the API timeout in a thread-safe manner.
// Lock-free: uses atomic.Int64.
func (t *TritonClientService) getAPITimeout() time.Duration {
	return time.Duration(t.apiTimeoutNanos.Load())
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
	timeout := t.getAPITimeout()
	if httpErr := t.httpClient.DoTimeout(requestObj, responseObj, timeout); httpErr != nil {
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
	timeout := t.getAPITimeout()
	if httpErr := t.httpClient.DoTimeout(requestObj, responseObj, timeout); httpErr != nil {
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
		return nil, newGRPCError("infer", errors.New("grpc client is nil"))
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
		return nil, newGRPCError("infer", inferErr)
	}
	return modelInferResponse, nil
}

// httpError creates a structured HTTP error.
func httpError(op string, statusCode int, err error) error {
	if err == nil && statusCode == http.StatusOK {
		return nil
	}
	if err == nil {
		err = errors.New("unexpected status code")
	}
	return newHTTPError(op, statusCode, err)
}

// grpcError creates a structured gRPC error, returning nil if grpcErr is nil.
func grpcError(op string, grpcErr error) error {
	if grpcErr == nil {
		return nil
	}
	return newGRPCError(op, grpcErr)
}

// decodeError creates a structured decode error.
func decodeError(op string, isGRPC bool, err error) error {
	if isGRPC {
		return newGRPCError(op+".decode", err)
	}
	return newHTTPError(op+".decode", 0, err)
}

// httpGetAndDecode performs an HTTP GET request, checks the response, and decodes the JSON body into result.
func (t *TritonClientService) httpGetAndDecode(op, requestURL string, result any) error {
	apiResp, httpErr := t.makeHTTPGetRequestWithDoTimeout(requestURL)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return httpError(op, http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return httpError(op, apiResp.StatusCode(), httpErr)
	}
	if result != nil {
		if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), result); jsonDecodeErr != nil {
			return jsonDecodeErr
		}
	}
	return nil
}

// httpPostAndDecode performs an HTTP POST request, checks the response, and decodes the JSON body into result.
func (t *TritonClientService) httpPostAndDecode(op, requestURL string, reqBody []byte, result any) error {
	apiResp, httpErr := t.makeHTTPPostRequestWithDoTimeout(requestURL, reqBody)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return httpError(op, http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return httpError(op, apiResp.StatusCode(), httpErr)
	}
	if result != nil {
		if jsonDecodeErr := t.JSONDecoder(apiResp.Body(), result); jsonDecodeErr != nil {
			return jsonDecodeErr
		}
	}
	return nil
}

///////////////////////////////////////////// expose API below /////////////////////////////////////////////

// JsonMarshal Json Encoder. Thread-safe.
func (t *TritonClientService) JsonMarshal(v any) ([]byte, error) {
	t.mu.RLock()
	encoder := t.JSONEncoder
	t.mu.RUnlock()
	return encoder(v)
}

// JsonUnmarshal Json Decoder. Thread-safe.
func (t *TritonClientService) JsonUnmarshal(data []byte, v any) error {
	t.mu.RLock()
	decoder := t.JSONDecoder
	t.mu.RUnlock()
	return decoder(data, v)
}

// SetJSONEncoder set json encoder. Thread-safe.
//
// Deprecated: Use WithJSONEncoder option during client construction instead.
// Runtime mutation of the encoder is discouraged in concurrent scenarios.
func (t *TritonClientService) SetJSONEncoder(encoder utils.JSONMarshal) *TritonClientService {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.JSONEncoder = encoder
	return t
}

// SetJSONDecoder set json decoder. Thread-safe.
//
// Deprecated: Use WithJSONDecoder option during client construction instead.
// Runtime mutation of the decoder is discouraged in concurrent scenarios.
func (t *TritonClientService) SetJSONDecoder(decoder utils.JSONUnmarshal) *TritonClientService {
	t.mu.Lock()
	defer t.mu.Unlock()
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
	const op = "infer"
	// Check context cancellation
	if err := ctx.Err(); err != nil {
		return nil, newHTTPError(op, 0, err)
	}
	// get infer response.
	modelInferResponse, inferErr := t.makeHTTPPostRequestWithDoTimeout(
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+TritonAPIForModelVersionPrefix+pathEscape(modelVersion)+"/infer",
		requestBody)
	defer fasthttp.ReleaseResponse(modelInferResponse)

	if modelInferResponse == nil {
		return nil, httpError(op, http.StatusInternalServerError, errors.New("modelInferResponse is nil"))
	}

	if inferErr != nil || modelInferResponse.StatusCode() != fasthttp.StatusOK {
		if inferErr == nil {
			body := modelInferResponse.Body()
			if len(body) > 0 {
				inferErr = errors.New("triton error: " + string(body))
			} else {
				inferErr = errors.New("unexpected status code")
			}
		}
		return nil, httpError(op, modelInferResponse.StatusCode(), inferErr)
	}
	// decode Result.
	response, decodeErr := decoderFunc(modelInferResponse.Body(), params...)
	if decodeErr != nil {
		return nil, decodeError(op, false, decodeErr)
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
	const op = "infer"
	// Get infer response.
	modelInferResponse, inferErr := t.modelGRPCInfer(
		ctx, inferInputs, inferOutputs, rawInputs, modelName, modelVersion)
	if inferErr != nil {
		return nil, inferErr
	}
	// decode Result.
	response, decodeErr := decoderFunc(modelInferResponse, params...)
	if decodeErr != nil {
		return nil, decodeError(op, true, decodeErr)
	}
	return response, nil
}

// CheckServerAlive check server is alive.
func (t *TritonClientService) CheckServerAlive(ctx context.Context) (bool, error) {
	const op = "health.live"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		serverLiveResponse, serverAliveErr := t.grpcClient.ServerLive(ctx, &ServerLiveRequest{})
		if serverAliveErr != nil {
			return false, grpcError(op, serverAliveErr)
		}
		return serverLiveResponse.Live, nil
	}
	apiResp, httpErr := t.makeHTTPGetRequestWithDoTimeout(t.getServerURL() + TritonAPIForServerIsLive)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return false, httpError(op, http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return false, httpError(op, apiResp.StatusCode(), httpErr)
	}
	return true, nil
}

// CheckServerReady check server is ready.
func (t *TritonClientService) CheckServerReady(ctx context.Context) (bool, error) {
	const op = "health.ready"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		serverReadyResponse, serverReadyErr := t.grpcClient.ServerReady(ctx, &ServerReadyRequest{})
		if serverReadyErr != nil {
			return false, grpcError(op, serverReadyErr)
		}
		return serverReadyResponse.Ready, nil
	}
	apiResp, httpErr := t.makeHTTPGetRequestWithDoTimeout(t.getServerURL() + TritonAPIForServerIsReady)
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return false, httpError(op, http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return false, httpError(op, apiResp.StatusCode(), httpErr)
	}
	return true, nil
}

// CheckModelReady check model is ready.
func (t *TritonClientService) CheckModelReady(ctx context.Context, modelName, modelVersion string) (bool, error) {
	const op = "model.ready"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		modelReadyResponse, modelReadyErr := t.grpcClient.ModelReady(
			ctx, &ModelReadyRequest{Name: modelName, Version: modelVersion})
		if modelReadyErr != nil {
			return false, grpcError(op, modelReadyErr)
		}
		return modelReadyResponse.Ready, nil
	}
	apiResp, httpErr := t.makeHTTPGetRequestWithDoTimeout(
		t.getServerURL() + TritonAPIForModelPrefix + pathEscape(modelName) + TritonAPIForModelVersionPrefix + pathEscape(modelVersion) + "/ready")
	defer fasthttp.ReleaseResponse(apiResp)
	if apiResp == nil {
		return false, httpError(op, http.StatusInternalServerError, utils.ErrApiRespNil)
	}
	if httpErr != nil || apiResp.StatusCode() != fasthttp.StatusOK {
		return false, httpError(op, apiResp.StatusCode(), httpErr)
	}
	return true, nil
}

// ServerMetadata Get server metadata.
func (t *TritonClientService) ServerMetadata(ctx context.Context) (*ServerMetadataResponse, error) {
	const op = "server.metadata"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		serverMetadataResponse, serverMetaErr := t.grpcClient.ServerMetadata(ctx, &ServerMetadataRequest{})
		return serverMetadataResponse, grpcError(op, serverMetaErr)
	}
	serverMetadataResponse := new(ServerMetadataResponse)
	return serverMetadataResponse, t.httpGetAndDecode(op, t.getServerURL()+TritonAPIPrefix, serverMetadataResponse)
}

// ModelMetadataRequest Get model metadata.
func (t *TritonClientService) ModelMetadataRequest(ctx context.Context, modelName, modelVersion string) (*ModelMetadataResponse, error) {
	const op = "model.metadata"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		modelMetadataResponse, modelMetaErr := t.grpcClient.ModelMetadata(
			ctx, &ModelMetadataRequest{Name: modelName, Version: modelVersion})
		return modelMetadataResponse, grpcError(op, modelMetaErr)
	}
	modelMetadataResponse := new(ModelMetadataResponse)
	return modelMetadataResponse, t.httpGetAndDecode(op,
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+TritonAPIForModelVersionPrefix+pathEscape(modelVersion),
		modelMetadataResponse)
}

// ModelIndex Get model repo index.
func (t *TritonClientService) ModelIndex(ctx context.Context, repoName string, isReady bool) (*RepositoryIndexResponse, error) {
	const op = "repository.index"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		repositoryIndexResponse, modelIndexErr := t.grpcClient.RepositoryIndex(
			ctx, &RepositoryIndexRequest{RepositoryName: repoName, Ready: isReady})
		return repositoryIndexResponse, grpcError(op, modelIndexErr)
	}
	reqBody, jsonEncodeErr := t.JsonMarshal(&ModelIndexRequestHTTPObj{repoName, isReady})
	if jsonEncodeErr != nil {
		return nil, jsonEncodeErr
	}
	repositoryIndexResponse := new(RepositoryIndexResponse)
	if httpErr := t.httpPostAndDecode(op, t.getServerURL()+TritonAPIForRepoIndex, reqBody, &repositoryIndexResponse.Models); httpErr != nil {
		return nil, httpErr
	}
	return repositoryIndexResponse, nil
}

// ModelConfiguration Get model configuration.
func (t *TritonClientService) ModelConfiguration(ctx context.Context, modelName, modelVersion string) (*ModelConfigResponse, error) {
	const op = "model.config"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		modelConfigResponse, getModelConfigErr := t.grpcClient.ModelConfig(
			ctx, &ModelConfigRequest{Name: modelName, Version: modelVersion})
		return modelConfigResponse, grpcError(op, getModelConfigErr)
	}
	modelConfigResponse := new(ModelConfigResponse)
	return modelConfigResponse, t.httpGetAndDecode(op,
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+
			TritonAPIForModelVersionPrefix+pathEscape(modelVersion)+"/config", modelConfigResponse)
}

// ModelInferStats Get Model infer stats.
func (t *TritonClientService) ModelInferStats(ctx context.Context, modelName, modelVersion string) (*ModelStatisticsResponse, error) {
	const op = "model.stats"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		modelStatisticsResponse, getInferStatsErr := t.grpcClient.ModelStatistics(
			ctx, &ModelStatisticsRequest{Name: modelName, Version: modelVersion})
		return modelStatisticsResponse, grpcError(op, getInferStatsErr)
	}
	modelStatisticsResponse := new(ModelStatisticsResponse)
	return modelStatisticsResponse, t.httpGetAndDecode(op,
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+
			TritonAPIForModelVersionPrefix+pathEscape(modelVersion)+"/stats", modelStatisticsResponse)
}

// ModelLoadWithHTTP Load Model with http
// modelConfigBody ==>
// https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md#examples
func (t *TritonClientService) ModelLoadWithHTTP(ctx context.Context, modelName string, modelConfigBody []byte) (*RepositoryModelLoadResponse, error) {
	const op = "model.load"
	repositoryModelLoadResponse := new(RepositoryModelLoadResponse)
	return repositoryModelLoadResponse, t.httpPostAndDecode(op,
		t.getServerURL()+TritonAPIForRepoModelPrefix+pathEscape(modelName)+"/load", modelConfigBody, repositoryModelLoadResponse)
}

// ModelLoadWithGRPC Load model with gRPC.
func (t *TritonClientService) ModelLoadWithGRPC(ctx context.Context, repoName, modelName string, modelConfigBody map[string]*ModelRepositoryParameter) (*RepositoryModelLoadResponse, error) {
	const op = "model.load"
	if t.grpcClient == nil {
		return nil, newGRPCError(op, errors.New("grpc client is nil"))
	}
	ctx, cancel := t.ensureCtx(ctx)
	defer cancel()
	loadResponse, loadErr := t.grpcClient.RepositoryModelLoad(ctx, &RepositoryModelLoadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return loadResponse, grpcError(op, loadErr)
}

// ModelUnloadWithHTTP Unload model with http
// modelConfigBody if not is nil.
func (t *TritonClientService) ModelUnloadWithHTTP(ctx context.Context, modelName string, modelConfigBody []byte) (*RepositoryModelUnloadResponse, error) {
	const op = "model.unload"
	repositoryModelUnloadResponse := new(RepositoryModelUnloadResponse)
	return repositoryModelUnloadResponse, t.httpPostAndDecode(op,
		t.getServerURL()+TritonAPIForRepoModelPrefix+pathEscape(modelName)+"/unload", modelConfigBody, repositoryModelUnloadResponse)
}

// ModelUnloadWithGRPC Unload model with gRPC.
func (t *TritonClientService) ModelUnloadWithGRPC(ctx context.Context, repoName, modelName string, modelConfigBody map[string]*ModelRepositoryParameter) (*RepositoryModelUnloadResponse, error) {
	const op = "model.unload"
	if t.grpcClient == nil {
		return nil, newGRPCError(op, errors.New("grpc client is nil"))
	}
	ctx, cancel := t.ensureCtx(ctx)
	defer cancel()
	unloadResponse, unloadErr := t.grpcClient.RepositoryModelUnload(ctx, &RepositoryModelUnloadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return unloadResponse, grpcError(op, unloadErr)
}

// ShareCUDAMemoryStatus Get CUDA shared memory status.
func (t *TritonClientService) ShareCUDAMemoryStatus(ctx context.Context, regionName string) (*CudaSharedMemoryStatusResponse, error) {
	const op = "cuda.memory.status"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		cudaSharedMemoryStatusResponse, cudaStatusErr := t.grpcClient.CudaSharedMemoryStatus(
			ctx, &CudaSharedMemoryStatusRequest{Name: regionName})
		return cudaSharedMemoryStatusResponse, grpcError(op, cudaStatusErr)
	}
	cudaSharedMemoryStatusResponse := new(CudaSharedMemoryStatusResponse)
	return cudaSharedMemoryStatusResponse, t.httpGetAndDecode(op,
		t.getServerURL()+TritonAPIForCudaMemoryRegionPrefix+pathEscape(regionName)+"/status", cudaSharedMemoryStatusResponse)
}

// ShareSystemMemoryStatus Get system shared memory status.
func (t *TritonClientService) ShareSystemMemoryStatus(ctx context.Context, regionName string) (*SystemSharedMemoryStatusResponse, error) {
	const op = "system.memory.status"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		systemSharedMemoryStatusResponse, systemStatusErr := t.grpcClient.SystemSharedMemoryStatus(
			ctx, &SystemSharedMemoryStatusRequest{Name: regionName})
		return systemSharedMemoryStatusResponse, grpcError(op, systemStatusErr)
	}
	systemSharedMemoryStatusResponse := new(SystemSharedMemoryStatusResponse)
	return systemSharedMemoryStatusResponse, t.httpGetAndDecode(op,
		t.getServerURL()+TritonAPIForSystemMemoryRegionPrefix+pathEscape(regionName)+"/status", systemSharedMemoryStatusResponse)
}

// ShareCUDAMemoryRegister cuda share memory register.
func (t *TritonClientService) ShareCUDAMemoryRegister(ctx context.Context, regionName string, cudaRawHandle []byte, cudaDeviceID int64, byteSize uint64) (*CudaSharedMemoryRegisterResponse, error) {
	const op = "cuda.memory.register"
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
		return cudaSharedMemoryRegisterResponse, grpcError(op, registerErr)
	}
	reqBody, jsonEncodeErr := t.JsonMarshal(
		&CudaMemoryRegisterBodyHTTPObj{cudaRawHandle, cudaDeviceID, byteSize})
	if jsonEncodeErr != nil {
		return nil, jsonEncodeErr
	}
	cudaSharedMemoryRegisterResponse := new(CudaSharedMemoryRegisterResponse)
	return cudaSharedMemoryRegisterResponse, t.httpPostAndDecode(op,
		t.getServerURL()+TritonAPIForCudaMemoryRegionPrefix+pathEscape(regionName)+"/register", reqBody, cudaSharedMemoryRegisterResponse)
}

// ShareCUDAMemoryUnRegister cuda share memory unregister.
func (t *TritonClientService) ShareCUDAMemoryUnRegister(ctx context.Context, regionName string) (*CudaSharedMemoryUnregisterResponse, error) {
	const op = "cuda.memory.unregister"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		cudaSharedMemoryUnRegisterResponse, unRegisterErr := t.grpcClient.CudaSharedMemoryUnregister(
			ctx, &CudaSharedMemoryUnregisterRequest{Name: regionName})
		return cudaSharedMemoryUnRegisterResponse, grpcError(op, unRegisterErr)
	}
	cudaSharedMemoryUnregisterResponse := new(CudaSharedMemoryUnregisterResponse)
	return cudaSharedMemoryUnregisterResponse, t.httpPostAndDecode(op,
		t.getServerURL()+TritonAPIForCudaMemoryRegionPrefix+pathEscape(regionName)+"/unregister", nil, cudaSharedMemoryUnregisterResponse)
}

// ShareSystemMemoryRegister system share memory register.
func (t *TritonClientService) ShareSystemMemoryRegister(ctx context.Context, regionName, cpuMemRegionKey string, byteSize, cpuMemOffset uint64) (*SystemSharedMemoryRegisterResponse, error) {
	const op = "system.memory.register"
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
		return systemSharedMemoryRegisterResponse, grpcError(op, registerErr)
	}
	reqBody, jsonEncodeErr := t.JsonMarshal(
		&SystemMemoryRegisterBodyHTTPObj{cpuMemRegionKey, cpuMemOffset, byteSize})
	if jsonEncodeErr != nil {
		return nil, jsonEncodeErr
	}
	systemSharedMemoryRegisterResponse := new(SystemSharedMemoryRegisterResponse)
	return systemSharedMemoryRegisterResponse, t.httpPostAndDecode(op,
		t.getServerURL()+TritonAPIForSystemMemoryRegionPrefix+pathEscape(regionName)+"/register", reqBody, systemSharedMemoryRegisterResponse)
}

// ShareSystemMemoryUnRegister system share memory unregister.
func (t *TritonClientService) ShareSystemMemoryUnRegister(ctx context.Context, regionName string) (*SystemSharedMemoryUnregisterResponse, error) {
	const op = "system.memory.unregister"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		systemSharedMemoryUnRegisterResponse, unRegisterErr := t.grpcClient.SystemSharedMemoryUnregister(
			ctx, &SystemSharedMemoryUnregisterRequest{Name: regionName})
		return systemSharedMemoryUnRegisterResponse, grpcError(op, unRegisterErr)
	}
	systemSharedMemoryUnregisterResponse := new(SystemSharedMemoryUnregisterResponse)
	return systemSharedMemoryUnregisterResponse, t.httpPostAndDecode(op,
		t.getServerURL()+TritonAPIForSystemMemoryRegionPrefix+pathEscape(regionName)+"/unregister", nil, systemSharedMemoryUnregisterResponse)
}

// GetModelTracingSetting get model tracing setting.
func (t *TritonClientService) GetModelTracingSetting(ctx context.Context, modelName string) (*TraceSettingResponse, error) {
	const op = "trace.get"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		traceSettingResponse, getTraceSettingErr := t.grpcClient.TraceSetting(
			ctx, &TraceSettingRequest{ModelName: modelName})
		return traceSettingResponse, grpcError(op, getTraceSettingErr)
	}
	traceSettingResponse := new(TraceSettingResponse)
	return traceSettingResponse, t.httpGetAndDecode(op,
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+"/trace/setting", traceSettingResponse)
}

// SetModelTracingSetting set model tracing setting.
// Param: settingMap ==>
// https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_trace.md#trace-setting-response-json-object
func (t *TritonClientService) SetModelTracingSetting(
	ctx context.Context, modelName string, settingMap map[string]*TraceSettingRequest_SettingValue,
) (*TraceSettingResponse, error) {
	const op = "trace.set"
	if t.grpcClient != nil {
		ctx, cancel := t.ensureCtx(ctx)
		defer cancel()

		traceSettingResponse, setTraceSettingErr := t.grpcClient.TraceSetting(
			ctx, &TraceSettingRequest{ModelName: modelName, Settings: settingMap})
		return traceSettingResponse, grpcError(op, setTraceSettingErr)
	}
	reqBody, jsonEncodeErr := t.JsonMarshal(&TraceSettingRequestHTTPObj{settingMap})
	if jsonEncodeErr != nil {
		return nil, jsonEncodeErr
	}
	traceSettingResponse := new(TraceSettingResponse)
	return traceSettingResponse, t.httpPostAndDecode(op,
		t.getServerURL()+TritonAPIForModelPrefix+pathEscape(modelName)+"/trace/setting", reqBody, traceSettingResponse)
}

// SetSecondaryServerURL Set secondary server url
func (t *TritonClientService) SetSecondaryServerURL(serverURL string) {
	t.secondaryServerURL.Store(serverURL)
}

// SetAPIRequestTimeout Set API request timeout.
func (t *TritonClientService) SetAPIRequestTimeout(timeout time.Duration) {
	t.apiTimeoutNanos.Store(int64(timeout))
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

// newBaseClient creates a TritonClientService with default settings.
func newBaseClient() *TritonClientService {
	c := &TritonClientService{
		JSONEncoder: json.Marshal,
		JSONDecoder: json.Unmarshal,
		logger:      slog.Default(),
	}
	c.apiTimeoutNanos.Store(int64(DefaultHTTPClientReadTimeout))
	return c
}

// NewTritonClientWithOnlyHTTP init triton client with HTTP only.
//
// For HTTPS/TLS support, pass a URL with "https://" prefix and configure TLS
// on the fasthttp.Client:
//
//	tlsConfig := &tls.Config{RootCAs: certPool}
//	httpClient := &fasthttp.Client{TLSConfig: tlsConfig}
//	client := NewTritonClientWithOnlyHTTP("https://triton:8000", httpClient)
func NewTritonClientWithOnlyHTTP(uri string, httpClient *fasthttp.Client, opts ...ClientOption) *TritonClientService {
	client := newBaseClient()
	client.serverURL = uri
	client.applyOptions(opts)
	if client.httpClient == nil {
		client.setHTTPConnection(httpClient)
	}
	return client
}

// NewTritonClientWithOnlyGRPC init triton client with gRPC only.
//
// For TLS support, create the grpc.ClientConn with TLS credentials:
//
//	creds := credentials.NewTLS(&tls.Config{RootCAs: certPool})
//	conn, _ := grpc.Dial("triton:8001", grpc.WithTransportCredentials(creds))
//	client, _ := NewTritonClientWithOnlyGRPC(conn)
//
// For keepalive support, configure it on the grpc.ClientConn:
//
//	kaParams := keepalive.ClientParameters{Time: 10 * time.Second, Timeout: 3 * time.Second, PermitWithoutStream: true}
//	conn, _ := grpc.Dial("triton:8001", grpc.WithKeepaliveParams(kaParams))
//	client, _ := NewTritonClientWithOnlyGRPC(conn)
func NewTritonClientWithOnlyGRPC(grpcConn *grpc.ClientConn, opts ...ClientOption) (*TritonClientService, error) {
	if grpcConn == nil {
		return nil, newGRPCError("connect", errors.New("grpc connection is nil"))
	}
	client := newBaseClient()
	client.grpcConn = grpcConn
	client.grpcClient = NewGRPCInferenceServiceClient(grpcConn)
	client.applyOptions(opts)
	return client, nil
}

// NewTritonClientForAll init triton client with http and grpc.
func NewTritonClientForAll(httpServerURL string, httpClient *fasthttp.Client, grpcConn *grpc.ClientConn, opts ...ClientOption) *TritonClientService {
	client := newBaseClient()
	client.serverURL = httpServerURL
	client.applyOptions(opts)
	if grpcConn != nil {
		client.grpcConn = grpcConn
		client.grpcClient = NewGRPCInferenceServiceClient(grpcConn)
	}
	if client.httpClient == nil {
		client.setHTTPConnection(httpClient)
	}
	return client
}
