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
- TLS/SSL not yet supported (planned)

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
	"fmt"
	"time"

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
func testGenerateModelInferOutputRequest(params ...interface{}) []*nvidia_inferenceserver.ModelInferRequest_InferRequestedOutputTensor {
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
func testModerInferCallback(inferResponse interface{}, params ...interface{}) ([]interface{}, error) {
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

	// Service
	bertService, initErr := transformers.NewBertModelService(
		vocabPath, httpAddr, defaultHttpClient, defaultGRPCClient,
		testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback)
	if initErr != nil {
		panic(initErr)
	}
	bertService.SetChineseTokenize(false).SetMaxSeqLength(maxSeqLen)
	// infer
	inferResultV1, inferErr := bertService.ModelInfer([]string{"<Data>"}, "<Model Name>", "<Model Version>", 1*time.Second)
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

### Version

* version 2.1.0 - 2025
  * **Proto files updated** to latest upstream ([triton-inference-server/common](https://github.com/triton-inference-server/common/tree/main/protobuf))
  * **Bug fixes**: `setHTTPConnection` logic bug, `ModelIndex` interface signature, `ShareMemoryStatus` return type
  * **Performance**: pre-allocated byte buffers in `grpcSliceToLittleEndianByteSlice`
  * **API changes**: `ShareMemoryStatus` split into `ShareCUDAMemoryStatus` / `ShareSystemMemoryStatus`; interface return types changed from `interface{}` to concrete types
  * **Tests**: added 20+ new unit tests and benchmarks
  * **Tooling**: added `Makefile`, fixed GitHub Actions workflow naming

* version 2.0.5 - 2024/07/26
  * Remove timeout for `TritonService` interface, use `SetAPIRequestTimeout` instead.
  * Add new api for `SetAPIRequestTimeout`

* version 2.0.4 - 2024/07/09
  * Update `W2NER` input feature problem.(Missing `MaxSeqLength` config)
  * Code style fix. Reducing nil cases
  * Add `slice.StringSliceTruncatePrecisely` function for logic to handle [][] string data truncation.

* version 2.0.3 - 2024/07/08
  * Fix `w2ner.pieces2word` nil slice caused infer error.

* version 2.0.2 - 2024/05/27
  * Fix `generateGRPCRequest` missing tensor shape.
  * Update go.mod

* version 2.0.1 - 2024/03/06
  * Fix `2.0.0` cannot read `go.mod` successfully.

* version 2.0.0 - 2024/03/06
  * **No longer compatible with Go version 1.18, 1.19, 1.20** 
  * refactor `models` package and rename package from `bert` to `transformers`.
    * **Incompatible with previous versions, calls require simple modifications**
  * Add `W2NER` model(Based on Bert, but used for NER tasks)

* version 1.4.6 - 2023/07/27
  * remove `github.com/goccy/go-json` and set `encoding/json` to default json marshal/unmarshal.
  * add `JsonEncoder` and `JsonDecoder` API to adapt other json parser.

* version 1.4.5 - 2023/07/12
  * update go.mod
  * fix Chinese tokenizer error

* version 1.4.4 - 2023/07/11
  * tokenize Chinese-English-Number text with char mode for NER task.

* version 1.4.3 - 2023/06/29
  * fix miss `params` parameters in `models/bert/model.go`

* version 1.4.2 - 2023/06/05
  * add `SetSecondaryServerURL` API for some special test environment.

* version 1.4.0 - 2023/03/28
  * add some functions for utils and some consts

* version 1.3.9 - 2023/03/07
  * optimize a bit of performance on tokenizer.
  * do `golangci-lint` jobs

* version 1.3.8 - 2023/03/06
  * update `go.mod`

* version 1.3.7 - 2023/03/03
  * update grpc proto and grpc codes to compatible triton inference server 23.02

* version 1.3.6 - 2023/03/01
  * fix version error

* version 1.3.5 - 2023/03/01
  * fix http response callback error

* version 1.3.4 - 2023/02/15
  * fix grpc input order

* version 1.3.3 - 2023/02/10
  * add API to return word offsets

* version 1.3.2 - 2023/02/10
  * update API support more params before infer or after infer.

* version 1.3.1 - 2023/02/10
  * update `go.mod`

* version 1.3.0 - 2023/02/08
  * Remove deprecated API
  * update `go.mod`

* version 1.2.7 - 2023/02/06
  * add `GetModelInferIsGRPC`, `GetTokenizerIsChineseMode` API
  * update `README.md`
  * update `go.mod`

* version 1.2.6 - 2023/02/03
  * add `Bert` service to call `Triton Inference Server`
  * update `go.mod`

* version 1.2.5 - 2023/01/13
  * fix `makeHttpGetRequestWithDoTimeout` error Method.

* version 1.2.4 - 2023/01/12
  * fix `GOPROXY` can not get `go.mod`

* version 1.2.3 - 2023/01/12
  * update grpc connection code
  * update connection api
  * add some const for uri
  * add some test code
  * [API Update]split share system/cuda memory api
  * [API Update]add three api for client initialize
  * [API Update]update model load/unload api
  * [API Update]remove `isGRPC` parameter instead of determine `grpcClient` is nil or not.
  * [API Update]decodeFunc return `[]interface{}` instead of `interface{}` for support batch request.

* version 1.2.2 - 2022/11/14
  * fix empty request/response pool

* version 1.1.8 - 2022/11/09
  * update grpc proto base on 22.07 [protobuf](https://github.com/triton-inference-server/common/tree/r22.07/protobuf)
  * update `nvidia_inferenceser` package for grpc service
  * use `github.com/goccy/go-json` instead of `github.com/bytedance/sonic`, because it will make memory pool larger than `goccy/go-json` with same QPS.
  * remove use `fiber` client make http request.
  * update `go.mod`

* version 1.1.7
  * ~~use `bytedance/sonic` instead of `encoding/json`~~
  * use `errors.New` instead of `fmt.Errorf`
  * remove `fmt` package usage
  * update `go.mod`

* version 1.1.2
  * update go.mod

* version 1.0.0
  * Implement about 90% API of Triton Inference Server HTTP/GRPC Protocol
