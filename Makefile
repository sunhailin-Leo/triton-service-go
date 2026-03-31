.PHONY: all test bench lint proto clean vet fmt help

# Go parameters
GOCMD      := go
GOTEST     := $(GOCMD) test
GOBUILD    := $(GOCMD) build
GOVET      := $(GOCMD) vet
GOFMT      := gofmt
GOMOD      := $(GOCMD) mod
MODULE     := github.com/sunhailin-Leo/triton-service-go/v2

# Proto parameters
PROTO_DIR  := proto
PROTO_OUT  := nvidia_inferenceserver

# Default target
all: fmt vet lint test

## help: Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        Run fmt, vet, lint, and test (default)"
	@echo "  test       Run all unit tests with race detector"
	@echo "  bench      Run all benchmarks"
	@echo "  lint       Run golangci-lint"
	@echo "  vet        Run go vet"
	@echo "  fmt        Run gofmt on all Go files"
	@echo "  proto      Regenerate protobuf Go stubs"
	@echo "  tidy       Run go mod tidy"
	@echo "  clean      Clean build cache and test cache"
	@echo "  check      Run all CI checks (fmt, vet, lint, test, bench)"
	@echo "  coverage   Run tests with coverage report"
	@echo "  vulncheck  Run govulncheck for vulnerability scanning"
	@echo ""

## test: Run all unit tests with race detector
test:
	$(GOTEST) ./... -v -race -count=1

## bench: Run all benchmarks
bench:
	$(GOTEST) ./... -benchmem -run=^$$ -bench .

## lint: Run golangci-lint
lint:
	@command -v golangci-lint >/dev/null 2>&1 || { echo "golangci-lint not found, install: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"; exit 1; }
	golangci-lint run --enable=nolintlint,gochecknoinits,bodyclose,gocritic --verbose

## vet: Run go vet
vet:
	$(GOVET) ./...

## fmt: Run gofmt on all Go files
fmt:
	$(GOFMT) -w .

## proto: Regenerate protobuf Go stubs
proto:
	@command -v protoc >/dev/null 2>&1 || { echo "protoc not found, install: https://grpc.io/docs/protoc-installation/"; exit 1; }
	@command -v protoc-gen-go >/dev/null 2>&1 || { echo "protoc-gen-go not found, install: go install google.golang.org/protobuf/cmd/protoc-gen-go@latest"; exit 1; }
	@command -v protoc-gen-go-grpc >/dev/null 2>&1 || { echo "protoc-gen-go-grpc not found, install: go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest"; exit 1; }
	mkdir -p $(PROTO_OUT)
	cd $(PROTO_DIR) && protoc -I . \
		--go_out=../$(PROTO_OUT) --go_opt=paths=source_relative \
		--go-grpc_out=../$(PROTO_OUT) --go-grpc_opt=paths=source_relative \
		model_config.proto grpc_service.proto health.proto

## tidy: Run go mod tidy
tidy:
	$(GOMOD) tidy

## clean: Clean build cache, test cache, and generated files
clean:
	$(GOCMD) clean -cache -testcache
	rm -f coverage.out coverage.html

## check: Run all CI checks (fmt, vet, lint, test, bench)
check: fmt vet lint test bench

## coverage: Run tests with coverage report
coverage:
	$(GOTEST) ./... -coverprofile=coverage.out -covermode=atomic
	$(GOCMD) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

## vulncheck: Run govulncheck for vulnerability scanning
vulncheck:
	@command -v govulncheck >/dev/null 2>&1 || { echo "govulncheck not found, install: go install golang.org/x/vuln/cmd/govulncheck@latest"; exit 1; }
	govulncheck ./...
