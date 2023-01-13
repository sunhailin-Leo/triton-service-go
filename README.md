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

---

### Version

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