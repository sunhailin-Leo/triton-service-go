.PHONY: all test bench lint proto clean vet fmt help fuzz fuzz-utils fuzz-transformers bench-utils bench-transformers bench-nvidia coverage-unit fuzz-time

# Go parameters
GOCMD      := go
GOTEST     := $(GOCMD) test
GOBUILD    := $(GOCMD) build
GOVET      := $(GOCMD) vet
GOFMT      := gofmt
GOMOD      := $(GOCMD) mod
MODULE     := github.com/sunhailin-Leo/triton-service-go/v2

# Fuzz parameters
FUZZ_TIME  ?= 15s
FUZZ_COUNT ?= 50

# Proto parameters
PROTO_OUT  := nvidia_inferenceserver

# Default target
all: fmt vet lint test

## help: Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all                Run fmt, vet, lint, and test (default)"
	@echo "  test               Run all unit tests with race detector"
	@echo "  bench              Run all benchmarks"
	@echo "  bench-utils        Run benchmarks for utils package"
	@echo "  bench-transformers Run benchmarks for transformers package"
	@echo "  bench-nvidia       Run benchmarks for nvidia_inferenceserver package"
	@echo "  fuzz               Run all fuzz tests (default FUZZ_TIME=15s)"
	@echo "  fuzz-utils         Run fuzz tests for utils package"
	@echo "  fuzz-transformers  Run fuzz tests for transformers package"
	@echo "  fuzz-time          Run fuzz tests with custom duration (e.g., make fuzz-time FUZZ_TIME=30s)"
	@echo "  lint               Run golangci-lint"
	@echo "  vet                Run go vet"
	@echo "  fmt                Run gofmt on all Go files"
	@echo "  proto              Download upstream proto files and regenerate Go stubs"
	@echo "  tidy               Run go mod tidy"
	@echo "  clean              Clean build cache and test cache"
	@echo "  check              Run all CI checks (fmt, vet, lint, test, bench)"
	@echo "  coverage           Run tests with coverage report"
	@echo "  coverage-unit      Run unit tests with per-function coverage report"
	@echo "  vulncheck          Run govulncheck for vulnerability scanning"
	@echo ""
	@echo "Fuzz Parameters:"
	@echo "  FUZZ_TIME=$(FUZZ_TIME)   Duration for each fuzz test (default: 15s)"
	@echo "  FUZZ_COUNT=$(FUZZ_COUNT) Number of iterations for fuzz seed corpus (default: 50)"
	@echo ""

## test: Run all unit tests with race detector
test:
	$(GOTEST) ./... -v -race -count=1

## bench: Run all benchmarks
bench:
	$(GOTEST) ./... -benchmem -run=^$$ -bench .

## bench-utils: Run benchmarks for utils package
bench-utils:
	$(GOTEST) ./utils/... -benchmem -run=^$$ -bench .

## bench-transformers: Run benchmarks for transformers package
bench-transformers:
	$(GOTEST) ./models/transformers/... -benchmem -run=^$$ -bench .

## bench-nvidia: Run benchmarks for nvidia_inferenceserver package
bench-nvidia:
	$(GOTEST) ./nvidia_inferenceserver/... -benchmem -run=^$$ -bench .

## fuzz: Run all fuzz tests
fuzz: fuzz-utils fuzz-transformers

## fuzz-utils: Run fuzz tests for utils package
fuzz-utils:
	@echo "=== Fuzz testing utils package (FUZZ_TIME=$(FUZZ_TIME)) ==="
	@for func in FuzzIsWhitespace FuzzIsControl FuzzIsPunctuation FuzzIsChinese FuzzIsChineseOrNumber FuzzClean FuzzPadChinese FuzzStripAccentsAndLower FuzzSplitPunctuation FuzzBinaryFilter FuzzBinaryToSlice FuzzBinaryToSlice_INT32_Roundtrip FuzzBinaryToSlice_FP32_Roundtrip FuzzBinaryToSlice_FP64_Roundtrip FuzzBinaryToSlice_INT64_Roundtrip; do \
		echo "--- Running $$func ---"; \
		$(GOTEST) ./utils/... -fuzz=$$func -fuzztime=$(FUZZ_TIME) -run=^$$ || exit 1; \
	done

## fuzz-transformers: Run fuzz tests for transformers package
fuzz-transformers:
	@echo "=== Fuzz testing transformers package (FUZZ_TIME=$(FUZZ_TIME)) ==="
	@for func in FuzzTokenize FuzzTokenizeChinese FuzzTokenizeChineseCharMode FuzzWordPieceTokenize FuzzPadChineseDirect FuzzCleanAndPadChineseWithWhiteSpace FuzzCleanDirect FuzzSplitPunctuationDirect FuzzVocabLongestSubstring; do \
		echo "--- Running $$func ---"; \
		$(GOTEST) ./models/transformers/... -fuzz=$$func -fuzztime=$(FUZZ_TIME) -run=^$$ || exit 1; \
	done

## fuzz-time: Run fuzz tests with custom duration
fuzz-time:
	@echo "=== Fuzz testing all packages (FUZZ_TIME=$(FUZZ_TIME)) ==="
	@for func in FuzzIsWhitespace FuzzIsControl FuzzIsPunctuation FuzzIsChinese FuzzIsChineseOrNumber FuzzClean FuzzPadChinese FuzzStripAccentsAndLower FuzzSplitPunctuation FuzzBinaryFilter FuzzBinaryToSlice FuzzBinaryToSlice_INT32_Roundtrip FuzzBinaryToSlice_FP32_Roundtrip FuzzBinaryToSlice_FP64_Roundtrip FuzzBinaryToSlice_INT64_Roundtrip; do \
		echo "--- Running utils/$$func ---"; \
		$(GOTEST) ./utils/... -fuzz=$$func -fuzztime=$(FUZZ_TIME) -run=^$$ || exit 1; \
	done
	@for func in FuzzTokenize FuzzTokenizeChinese FuzzTokenizeChineseCharMode FuzzWordPieceTokenize FuzzPadChineseDirect FuzzCleanAndPadChineseWithWhiteSpace FuzzCleanDirect FuzzSplitPunctuationDirect FuzzVocabLongestSubstring; do \
		echo "--- Running transformers/$$func ---"; \
		$(GOTEST) ./models/transformers/... -fuzz=$$func -fuzztime=$(FUZZ_TIME) -run=^$$ || exit 1; \
	done

## lint: Run golangci-lint
lint:
	@command -v golangci-lint >/dev/null 2>&1 || { echo "golangci-lint not found, install: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"; exit 1; }
	golangci-lint run --enable=nolintlint,bodyclose,gocritic --verbose

## vet: Run go vet
vet:
	$(GOVET) ./...

## fmt: Run gofmt on all Go files
fmt:
	$(GOFMT) -w .

## proto: Download upstream proto files and regenerate Go stubs
proto:
	cd $(PROTO_OUT) && go generate ./...

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

## coverage-unit: Run unit tests with per-function coverage report
coverage-unit:
	$(GOTEST) ./utils/... ./models/... ./nvidia_inferenceserver/... -coverprofile=coverage.out -covermode=atomic
	@echo "--- Coverage by function ---"
	$(GOCMD) tool cover -func=coverage.out

## vulncheck: Run govulncheck for vulnerability scanning
vulncheck:
	@command -v govulncheck >/dev/null 2>&1 || { echo "govulncheck not found, install: go install golang.org/x/vuln/cmd/govulncheck@latest"; exit 1; }
	govulncheck ./...
