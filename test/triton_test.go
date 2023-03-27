package test

import (
	"log"
	"testing"
	"time"

	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"
)

func TestTritonHTTPClientForCheckModelReady(_ *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientForAll(
		"<Your Triton HTTP Host>:<Your Triton HTTP Port>", &fasthttp.Client{}, nil)
	isReady, err := srv.CheckModelReady(
		"<Your Model Name>", "<Your Model Version>", 1*time.Second)
	if err != nil {
		panic(err)
	}
	log.Println(isReady)
}

func TestTritonGRPCClientForCheckModelReady(_ *testing.T) {
	defaultGRPCClient, grpcErr := grpc.Dial("<Your Triton GRPC Host>:<Your Triton GRPC Port>",
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if grpcErr != nil {
		panic(grpcErr)
	}
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(defaultGRPCClient)
	isReady, err := srv.CheckModelReady(
		"<Your Model Name>", "<Your Model Version>", 1*time.Second)
	if err != nil {
		panic(err)
	}
	log.Println(isReady)
}

func TestTritonHTTPClientInit(_ *testing.T) {
	client := &fasthttp.Client{}
	trtClient := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(
		"<Your Triton HTTP Host>:<Your Triton HTTP Port>", client)
	isReady, err := trtClient.CheckServerReady(1 * time.Second)
	if err != nil {
		panic(err)
	}
	log.Println(isReady)
}

func TestTritonGRPCClientInit(_ *testing.T) {
	grpcConn, err := grpc.Dial(
		"<Your Triton GRPC Host>:<Your Triton GRPC Port>",
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		panic(err)
	}
	trtClient := nvidia_inferenceserver.NewTritonClientWithOnlyGRPC(grpcConn)
	isReady, err := trtClient.CheckServerReady(1 * time.Second)
	if err != nil {
		panic(err)
	}
	log.Println(isReady)
}

func TestTritonAllClientInit(_ *testing.T) {
	grpcConn, err := grpc.Dial(
		"<Your Triton GRPC Host>:<Your Triton GRPC Port>",
		grpc.WithTransportCredentials(insecure.NewCredentials()))
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
	log.Println(isReady)
}

func TestGetTritonModelConfig(_ *testing.T) {
	srv := nvidia_inferenceserver.NewTritonClientWithOnlyHTTP(
		"<Your Triton GRPC Host>:<Your Triton GRPC Port>", &fasthttp.Client{})
	modelConfig, err := srv.ModelConfiguration(
		"<Your Model Name>", "<Your Model Version>", 1*time.Second)
	if err != nil {
		panic(err)
	}
	log.Println(modelConfig)
}
