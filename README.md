# triton-service-go
Unofficial golang package for the Triton Inference Server(https://github.com/triton-inference-server/server)
Triton Inference Server - Golang API

---

### Attention

* ~~Currently only supported up to version 21.05 (Triton Inference Server), but compatible with 9x% of 22.07 versions~~
* Currently supported base on version 22.07

---

### Feature

* Support HTTP/GRPC
* Easy to use it
* Maybe High Performance
* Implement 98% API of Triton Inference Server HTTP/GRPC Protocol
* Cannot Support TLS/SSL now (https/grpc secure mode)...(will test it soon...)

--- 

### Usage

* Download
```shell
go get -u github.com/sunhailin-Leo/triton-service-go
```

* Example for `Bert` Model
```go
package main

import (
    "fmt"
	
    "github.com/valyala/fasthttp"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"

    "github.com/sunhailin-Leo/triton-service-go/models/bert"
    "github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"
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
func testGenerateModelInferRequest(batchSize, maxSeqLength int) []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor {
  return []*nvidia_inferenceserver.ModelInferRequest_InferInputTensor{
      {
            Name:     tBertModelSegmentIdsKey,
            Datatype: tBertModelSegmentIdsDataType,
            Shape:    []int64{int64(batchSize), int64(maxSeqLength)},
      },
      {
            Name:     tBertModelInputIdsKey,
            Datatype: tBertModelInputIdsDataType,
            Shape:    []int64{int64(batchSize), int64(maxSeqLength)},
      },
      {
            Name:     tBertModelInputMaskKey,
            Datatype: tBertModelInputMaskDataType,
            Shape:    []int64{int64(batchSize), int64(maxSeqLength)},
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
    defaultGRPCClient, grpcErr := grpc.Dial(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if grpcErr != nil {
        panic(grpcErr)
    }
    
    // Service
    bertService, initErr := bert.NewModelService(
    vocabPath, httpAddr, defaultHttpClient, defaultGRPCClient,
    testGenerateModelInferRequest, testGenerateModelInferOutputRequest, testModerInferCallback)
    if initErr != nil {
        panic(initErr)
    }
    bertService = bertService.SetChineseTokenize().SetMaxSeqLength(maxSeqLen)
    // infer
    inferResultV1, inferErr := bertService.ModelInfer([]string{"<Data>"}, "<Model Name>", "<Model Version>", 1*time.Second)
    if inferErr != nil {
        panic(inferErr)
    }
    println(inferResultV1)
}

```

---

### Version

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
