//go:build ignore

package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// ---- Configuration ----
// Default pinned version from triton-inference-server/common.
// Update this to sync to a newer version of the upstream proto definitions.
const defaultVersion = "r26.04"

const upstreamRepo = "triton-inference-server/common"

var protoFiles = []string{
	"grpc_service.proto",
	"health.proto",
	"model_config.proto",
}

const goPackage = "./nvidia_inferenceserver"

func main() {
	version := flag.String("version", defaultVersion, "upstream version (tag, branch, or commit SHA)")
	protoDir := flag.String("proto-dir", "", "directory to store downloaded .proto files (required)")
	outDir := flag.String("out-dir", "", "directory to write generated Go stubs (required)")
	flag.Parse()

	if *protoDir == "" || *outDir == "" {
		fmt.Fprintln(os.Stderr, "Error: --proto-dir and --out-dir are required")
		flag.Usage()
		os.Exit(1)
	}

	// Resolve to absolute paths for clarity in output.
	absProtoDir, _ := filepath.Abs(*protoDir)
	absOutDir, _ := filepath.Abs(*outDir)

	fmt.Printf("==> Downloading proto files from %s at version %s\n", upstreamRepo, *version)

	// Download proto files.
	for _, pf := range protoFiles {
		url := fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/protobuf/%s", upstreamRepo, *version, pf)
		dest := filepath.Join(absProtoDir, pf)
		fmt.Printf("    Downloading %s ...\n", pf)
		if err := downloadFile(url, dest); err != nil {
			fmt.Fprintf(os.Stderr, "Error downloading %s: %v\n", pf, err)
			os.Exit(1)
		}
	}

	fmt.Println("==> Proto files downloaded successfully")

	// Inject go_package option.
	// Upstream proto files do not include go_package, so we inject it after each "package" declaration.
	fmt.Println("==> Injecting go_package option ...")
	for _, pf := range protoFiles {
		dest := filepath.Join(absProtoDir, pf)
		if err := injectGoPackage(dest, goPackage); err != nil {
			fmt.Fprintf(os.Stderr, "Error injecting go_package into %s: %v\n", pf, err)
			os.Exit(1)
		}
		fmt.Printf("    Injected go_package into %s\n", pf)
	}

	// Check protoc toolchain.
	if _, err := exec.LookPath("protoc"); err != nil {
		fmt.Fprintln(os.Stderr, "Error: protoc not found, install: https://grpc.io/docs/protoc-installation/")
		os.Exit(1)
	}
	if _, err := exec.LookPath("protoc-gen-go"); err != nil {
		fmt.Fprintln(os.Stderr, "Error: protoc-gen-go not found, install: go install google.golang.org/protobuf/cmd/protoc-gen-go@latest")
		os.Exit(1)
	}
	if _, err := exec.LookPath("protoc-gen-go-grpc"); err != nil {
		fmt.Fprintln(os.Stderr, "Error: protoc-gen-go-grpc not found, install: go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest")
		os.Exit(1)
	}

	// Generate Go stubs.
	fmt.Println("==> Generating Go stubs ...")
	if err := os.MkdirAll(absOutDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	protocArgs := []string{
		"-I", absProtoDir,
		"--go_out=" + absOutDir, "--go_opt=paths=source_relative",
		"--go-grpc_out=" + absOutDir, "--go-grpc_opt=paths=source_relative",
	}
	protocArgs = append(protocArgs, protoFiles...)

	cmd := exec.Command("protoc", protocArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running protoc: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("==> Go stubs generated in %s\n", absOutDir)
	fmt.Printf("==> Done. Pinned version: %s\n", *version)
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("HTTP GET failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP status %d", resp.StatusCode)
	}

	f, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	if _, err := io.Copy(f, resp.Body); err != nil {
		return fmt.Errorf("write file: %w", err)
	}
	return nil
}

func injectGoPackage(path, pkg string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	content := string(data)

	// Skip if go_package already present.
	if strings.Contains(content, "option go_package") {
		return nil
	}

	// Insert go_package after the "package ..." line.
	injection := fmt.Sprintf("option go_package = \"%s\";", pkg)
	lines := strings.Split(content, "\n")
	var result []string
	inserted := false
	for _, line := range lines {
		result = append(result, line)
		if !inserted && strings.HasPrefix(line, "package ") {
			result = append(result, injection)
			inserted = true
		}
	}

	if !inserted {
		return fmt.Errorf("no 'package' declaration found in %s", path)
	}

	return os.WriteFile(path, []byte(strings.Join(result, "\n")), 0o644)
}
