# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v2.1.0] - 2026-03-31

### Added

- Comprehensive unit tests across all packages with high coverage (utils 98.8%, models 80.5%, transformers 48.0%, nvidia_inferenceserver 2.1%)
- 34 benchmark tests for core hot paths (tokenizer, slice operations, text processing, gRPC encoding)
- `Makefile` with standardized dev targets: `test`, `bench`, `coverage`, `lint`, `vet`, `fmt`, `proto`, `vulncheck`, `check`, `clean`
- GitHub Actions coverage job with Codecov upload
- Go 1.26.x support in CI matrix
- `scanner.Err()` handling in vocab file loading for better error detection
- `GrpcSliceToLittleEndianByteSlice` exported wrapper for testing gRPC tensor encoding

### Changed

- Proto files synced to latest upstream ([triton-inference-server/common](https://github.com/triton-inference-server/common/tree/main/protobuf))
- Regenerated Go protobuf stubs with split `*_grpc.pb.go` files (modern `--go_out`/`--go-grpc_out` flags)
- Interface return types changed from `interface{}` to concrete proto response types (e.g., `GetModelConfig` returns `*ModelConfigResponse`)
- Callback types updated to `...any` for cleaner variadic signatures
- `ShareMemoryStatus` split into `ShareCUDAMemoryStatus` / `ShareSystemMemoryStatus`
- `Flatten2DSlice` optimized with capacity preallocation
- `grpcSliceToLittleEndianByteSlice` optimized with pre-allocated byte buffers
- GitHub Actions workflows: bumped Go versions, upgraded action versions (`actions/setup-go@v6`, `actions/checkout@v6`, `golangci/golangci-lint-action@v9`)
- Simplified vulncheck workflow to scan entire project

### Fixed

- `setHTTPConnection` logic bug when HTTP client was nil
- `ModelIndex` interface signature mismatch
- Workflow file typo: `sercurity.yml` → `security.yml`
- gRPC deprecated API: `grpc.Dial` → `grpc.NewClient`
- Resource leak: added `t.Cleanup` for gRPC connections in tests

### Dependencies

- `google.golang.org/grpc` → v1.79.3
- `google.golang.org/protobuf` → v1.36.10
- `github.com/valyala/fasthttp` → v1.69.0
- `golang.org/x/text` → v0.34.0

## [v2.0.6] - 2025

### Added

- Configurable `doLowerCase` option for tokenizers
- Tests for `doLowerCase` behavior

### Changed

- Updated dependencies and minimum Go version to 1.24.x+

## [v2.0.5] - 2024-07-26

### Changed

- Removed timeout parameter from `TritonService` interface methods
- Added `SetAPIRequestTimeout` API for configuring request timeouts

## [v2.0.4] - 2024-07-09

### Fixed

- `W2NER` input feature missing `MaxSeqLength` config
- Code style improvements, reducing nil cases

### Added

- `slice.StringSliceTruncatePrecisely` function for precise `[][]string` data truncation

## [v2.0.3] - 2024-07-08

### Fixed

- `w2ner.pieces2word` nil slice causing infer error

## [v2.0.2] - 2024-05-27

### Fixed

- `generateGRPCRequest` missing tensor shape
- Updated `go.mod`

## [v2.0.1] - 2024-03-06

### Fixed

- v2.0.0 `go.mod` read failure

## [v2.0.0] - 2024-03-06

### Breaking Changes

- **No longer compatible with Go 1.18, 1.19, 1.20**
- Refactored `models` package: renamed `bert` package to `transformers` (requires call-site updates)

### Added

- `W2NER` model service (based on BERT, for NER tasks)

## [v1.4.6] - 2023-07-27

### Changed

- Removed `github.com/goccy/go-json`, set `encoding/json` as default
- Added `JsonEncoder` and `JsonDecoder` API for pluggable JSON parsers

## [v1.4.5] - 2023-07-12

### Fixed

- Chinese tokenizer error

## [v1.4.4] - 2023-07-11

### Added

- Chinese-English-Number text tokenization with char mode for NER tasks

## [v1.4.3] - 2023-06-29

### Fixed

- Missing `params` parameters in `models/bert/model.go`

## [v1.4.2] - 2023-06-05

### Added

- `SetSecondaryServerURL` API for special test environments

## [v1.4.0] - 2023-03-28

### Added

- Utility functions and constants

## [v1.3.9] - 2023-03-07

### Changed

- Optimized tokenizer performance
- Applied `golangci-lint` fixes

## [v1.3.8] - 2023-03-06

### Changed

- Updated `go.mod`

## [v1.3.7] - 2023-03-03

### Changed

- Updated gRPC proto and codes for Triton Inference Server 23.02 compatibility

## [v1.3.6] - 2023-03-01

### Fixed

- Version error

## [v1.3.5] - 2023-03-01

### Fixed

- HTTP response callback error

## [v1.3.4] - 2023-02-15

### Fixed

- gRPC input order

## [v1.3.3] - 2023-02-10

### Added

- API to return word offsets

## [v1.3.2] - 2023-02-10

### Changed

- Updated API to support more params before/after infer

## [v1.3.1] - 2023-02-10

### Changed

- Updated `go.mod`

## [v1.3.0] - 2023-02-08

### Changed

- Removed deprecated APIs
- Updated `go.mod`

## [v1.2.7] - 2023-02-06

### Added

- `GetModelInferIsGRPC`, `GetTokenizerIsChineseMode` APIs

## [v1.2.6] - 2023-02-03

### Added

- `Bert` service for calling Triton Inference Server

## [v1.2.5] - 2023-01-13

### Fixed

- `makeHttpGetRequestWithDoTimeout` incorrect method

## [v1.2.4] - 2023-01-12

### Fixed

- `GOPROXY` cannot get `go.mod`

## [v1.2.3] - 2023-01-12

### Changed

- Updated gRPC connection code and connection API
- Split share system/CUDA memory API
- Added three APIs for client initialization
- Updated model load/unload API
- Removed `isGRPC` parameter (determine by `grpcClient` nil check)
- `decodeFunc` returns `[]interface{}` for batch request support

## [v1.2.2] - 2022-11-14

### Fixed

- Empty request/response pool

## [v1.1.8] - 2022-11-09

### Changed

- Updated gRPC proto based on Triton 22.07 protobuf
- Replaced `github.com/goccy/go-json` with `encoding/json` (lower memory usage)
- Removed `fiber` HTTP client

## [v1.1.7]

### Changed

- Use `errors.New` instead of `fmt.Errorf`
- Removed `fmt` package usage

## [v1.0.0]

### Added

- Initial release: ~90% API coverage of Triton Inference Server HTTP/gRPC protocol

[v2.1.0]: https://github.com/sunhailin-Leo/triton-service-go/compare/v2.0.6...v2.1.0
[v2.0.6]: https://github.com/sunhailin-Leo/triton-service-go/compare/v2.0.5...v2.0.6
[v2.0.5]: https://github.com/sunhailin-Leo/triton-service-go/compare/v2.0.4...v2.0.5
[v2.0.4]: https://github.com/sunhailin-Leo/triton-service-go/compare/v2.0.3...v2.0.4
[v2.0.3]: https://github.com/sunhailin-Leo/triton-service-go/compare/v2.0.2...v2.0.3
[v2.0.2]: https://github.com/sunhailin-Leo/triton-service-go/compare/v2.0.1...v2.0.2
[v2.0.1]: https://github.com/sunhailin-Leo/triton-service-go/compare/v2.0.0...v2.0.1
[v2.0.0]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.4.6...v2.0.0
[v1.4.6]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.4.5...v1.4.6
[v1.4.5]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.4.4...v1.4.5
[v1.4.4]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.4.3...v1.4.4
[v1.4.3]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.4.2...v1.4.3
[v1.4.2]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.4.0...v1.4.2
[v1.4.0]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.3.9...v1.4.0
[v1.3.9]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.3.8...v1.3.9
[v1.3.8]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.3.7...v1.3.8
[v1.3.7]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.3.6...v1.3.7
[v1.3.6]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.3.5...v1.3.6
[v1.3.5]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.3.4...v1.3.5
[v1.3.4]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.3.3...v1.3.4
[v1.3.3]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.3.2...v1.3.3
[v1.3.2]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.3.1...v1.3.2
[v1.3.1]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.3.0...v1.3.1
[v1.3.0]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.2.7...v1.3.0
[v1.2.7]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.2.6...v1.2.7
[v1.2.6]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.2.5...v1.2.6
[v1.2.5]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.2.4...v1.2.5
[v1.2.4]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.2.3...v1.2.4
[v1.2.3]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.2.2...v1.2.3
[v1.2.2]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.1.8...v1.2.2
[v1.1.8]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.1.7...v1.1.8
[v1.1.7]: https://github.com/sunhailin-Leo/triton-service-go/compare/v1.0.0...v1.1.7
[v1.0.0]: https://github.com/sunhailin-Leo/triton-service-go/releases/tag/v1.0.0
