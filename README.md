# triton-service-go

Unofficial Golang SDK for [Triton Inference Server](https://github.com/triton-inference-server/server) — providing a complete HTTP/gRPC client for model inference, management, health checking, and shared memory operations.

[![Docs](https://pkg.go.dev/badge/github.com/sunhailin-Leo/triton-service-go)](https://pkg.go.dev/github.com/sunhailin-Leo/triton-service-go)
[![Report Card](https://goreportcard.com/badge/github.com/sunhailin-Leo/triton-service-go)](https://goreportcard.com/report/github.com/sunhailin-Leo/triton-service-go)

[![Benchmark](https://github.com/sunhailin-Leo/triton-service-go/actions/workflows/benchmark.yml/badge.svg)](https://github.com/sunhailin-Leo/triton-service-go/actions/workflows/benchmark.yml)
[![Lint Check](https://github.com/sunhailin-Leo/triton-service-go/actions/workflows/lint.yml/badge.svg)](https://github.com/sunhailin-Leo/triton-service-go/actions/workflows/lint.yml)
[![Security Check](https://github.com/sunhailin-Leo/triton-service-go/actions/workflows/security.yml/badge.svg)](https://github.com/sunhailin-Leo/triton-service-go/actions/workflows/security.yml)
[![Test](https://github.com/sunhailin-Leo/triton-service-go/actions/workflows/test.yml/badge.svg)](https://github.com/sunhailin-Leo/triton-service-go/actions/workflows/test.yml)
[![Vulnerability Check](https://github.com/sunhailin-Leo/triton-service-go/actions/workflows/vulncheck.yml/badge.svg)](https://github.com/sunhailin-Leo/triton-service-go/actions/workflows/vulncheck.yml)
[![Goproxy.cn](https://goproxy.cn/stats/github.com/sunhailin-Leo/triton-service-go/badges/download-count.svg)](https://goproxy.cn)

---

### Feature

> 🌟 **Go** ≥ 1.24 · **Triton Inference Server** ≥ 24.x (protobuf synced with [triton-inference-server/common](https://github.com/triton-inference-server/common/tree/main/protobuf))

- **HTTP + gRPC** dual-protocol support via a unified `TritonService` interface
- **High performance** — uses [fasthttp](https://github.com/valyala/fasthttp) for HTTP, pre-allocated byte buffers for gRPC tensor encoding
- **98% API coverage** of Triton Inference Server HTTP/gRPC protocol
- **Built-in BERT / W2NER model services** with WordPiece tokenizer
- **Pluggable JSON encoder/decoder** — swap in `sonic`, `go-json`, etc.
- **TLS/SSL support** — pass `https://` URLs for HTTP or configure `tls.Config` on gRPC connections
- **Structured errors** — `TritonError` type with `errors.Is`/`errors.As` support
- **Functional options** — `ClientOption` pattern for flexible client configuration
- **Observable** — optional `slog.Logger` injection via `WithLogger`

---

### Installation

```shell
go get -u github.com/sunhailin-Leo/triton-service-go/v2
```

---

### Quick Start

* Example for `Bert` Model
```go
package main

import (
	"context"
	"fmt"

    "github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
    "github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
    "github.com/valyala/fasthttp"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
)

const (
	tBertModelSegmentIdsKey                       string = "segment_ids"
	tBertModelSegmentIdsDataType                  string = "INT32"
	tBertModelInputIdsKey                         string = "input_ids"
	tBertModelInputIdsDataType                    string = "INT32"
	tBertModelInputMaskKey                        string = "input_mask"
	tBertModelInputMaskDataType                   string = "INT32"
	tBertModelOutputProbabilitiesKey              string = "probability"
	tBertModelRespBodyOutputBinaryDataKey         string = "binary_data"
	tBertModelRespBodyOutputClassificationDataKey string = "classification"
)

// testGenerateModelInferRequest Triton Input
func testGenerateModelInferRequest() []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor {
	return []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor{
		{
			Name:     tBertModelSegmentIdsKey,
			Datatype: tBertModelSegmentIdsDataType,
		},
		{
			Name:     tBertModelInputIdsKey,
			Datatype: tBertModelInputIdsDataType,
		},
		{
			Name:     tBertModelInputMaskKey,
			Datatype: tBertModelInputMaskDataType,
		},
	}
}

// testGenerateModelInferOutputRequest Triton Output
func testGenerateModelInferOutputRequest(params ...any) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor {
	return []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor{
		{
			Name: tBertModelOutputProbabilitiesKey,
			Parameters: map[string]*nvidia_inferenceserver.InferParameter{
				tBertModelRespBodyOutputBinaryDataKey: {
					ParameterChoice: &nvidia_inferenceserver.InferParameter_BoolParam{BoolParam: false},
				},
				tBertModelRespBodyOutputClassificationDataKey: {
					ParameterChoice: &nvidia_inferenceserver.InferParameter_Int64Param{Int64Param: 1},
				},
			},
		},
	}
}

// testModerInferCallback infer call back (process model infer data)
func testModerInferCallback(inferResponse any, params ...any) ([]any, error) {
	fmt.Println(inferResponse)
	fmt.Println(params...)
	return nil, nil
}


func main() {
	vocabPath := "<Your Bert Vocab Path>"
	maxSeqLen := 48
	httpAddr := "<HTTP URL>"
	grpcAddr := "<GRPC URL>"
	defaultHttpClient := &fasthttp.Client{}
	defaultGRPCClient, grpcErr := grpc.NewClient(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if grpcErr != nil {
		panic(grpcErr)
	}

	// Service (Option pattern for configuration)
	bertService, initErr := transformers.NewBertModelService(
		vocabPath, httpAddr, defaultHttpClient, defaultGRPCClient,
		testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback,
		transformers.WithBertChineseTokenize(false),
		transformers.WithBertMaxSeqLength(maxSeqLen),
	)
	if initErr != nil {
		panic(initErr)
	}
	// infer
	inferResultV1, inferErr := bertService.ModelInfer(context.Background(), []string{"<Data>"}, "<Model Name>", "<Model Version>")
	if inferErr != nil {
		panic(inferErr)
	}
	println(inferResultV1)
}
```

---

### Development

This project provides a `Makefile` for common development tasks:

```shell
make help       # Show all available targets
make test       # Run all unit tests with race detector
make bench      # Run all benchmarks
make lint       # Run golangci-lint
make proto      # Regenerate protobuf Go stubs
make coverage   # Generate HTML coverage report
make vulncheck  # Run govulncheck
make check      # Run all CI checks (fmt + vet + lint + test + bench)
```

### Project Structure

```
├── nvidia_inferenceserver/   # Triton gRPC/HTTP client & generated protobuf stubs
├── models/
│   ├── base.go               # Base model interface
│   └── transformers/         # BERT / W2NER model services & WordPiece tokenizer
├── utils/                    # Utility functions (slice, time, text processing)
├── proto/                    # Protobuf source files (.proto)
├── test/                     # Unit tests & benchmarks
└── Makefile                  # Development workflow automation
```

---

### Latest Version

**v2.1.0** - 2026/03/31
- Proto files synced with latest upstream, regenerated Go stubs
- Bug fixes: `setHTTPConnection`, `ModelIndex` signature, `ShareMemoryStatus` return type
- Performance: pre-allocated byte buffers for gRPC encoding, `Flatten2DSlice` optimization
- Comprehensive unit tests (utils 98.8%, models 80.5%)
- Added `Makefile`, GitHub Actions CI improvements

For full version history, see [CHANGELOG.md](./CHANGELOG.md).
