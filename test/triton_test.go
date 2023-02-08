package test

import (
	"fmt"
	"testing"
	"time"

	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"
)

func TestTritonHTTPClientForCheckModelReady(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientForAll(
		"<Your Triton HTTP Host>:<Your Triton HTTP Port>", &fasthttp.Client{}, nil)
	isReady, err := srv.CheckModelReady("<Your Model Name>", "<Your Model Version>", 1*time.Second)
	if err != nil {
		panic(err)
	}
	fmt.Println(isReady)
}

func TestTritonGRPCClientForCheckModelReady(t *testing.T) {
	defaultGRPCClient, grpcErr := grpc.Dial("<Your Triton GRPC Host>:<Your Triton GRPC Port>",
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if grpcErr != nil {
		panic(grpcErr)
	}
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(defaultGRPCClient)
	isReady, err := srv.CheckModelReady("<Your Model Name>", "<Your Model Version>", 1*time.Second)
	if err != nil {
		panic(err)
	}
	fmt.Println(isReady)
}

func TestTritonHTTPClientInit(t *testing.T) {
	client := &fasthttp.Client{}
	trtClient := nvidia_inferenceserver.NewTritonClientWithOnlyHttp("<Your Triton HTTP Host>:<Your Triton HTTP Port>", client)
	isReady, err := trtClient.CheckServerReady(1 * time.Second)
	if err != nil {
		panic(err)
	}
	fmt.Println(isReady)
}

func TestTritonGRPCClientInit(t *testing.T) {
	grpcConn, err := grpc.Dial("<Your Triton GRPC Host>:<Your Triton GRPC Port>", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		panic(err)
	}
	trtClient := nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(grpcConn)
	isReady, err := trtClient.CheckServerReady(1 * time.Second)
	if err != nil {
		panic(err)
	}
	fmt.Println(isReady)
}

func TestTritonAllClientInit(t *testing.T) {
	grpcConn, err := grpc.Dial("<Your Triton GRPC Host>:<Your Triton GRPC Port>", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		panic(err)
	}
	httpClient := &fasthttp.Client{}
	trtClient := nvidia_inferenceserver.NewTritonClientForAll(
		"<Your Triton HTTP Host>:<Your Triton HTTP Port>", httpClient, grpcConn)
	isReady, err := trtClient.CheckServerReady(1 * time.Second)
	if err != nil {
		panic(err)
	}
	fmt.Println(isReady)
}

func TestGetTritonModelConfig(t *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHttp("<Your Triton GRPC Host>:<Your Triton GRPC Port>", &fasthttp.Client{})
	modelConfig, err := srv.ModelConfiguration("<Your Model Name>", "<Your Model Version>", 1*time.Second)
	if err != nil {
		panic(err)
	}
	fmt.Println(modelConfig)
}
