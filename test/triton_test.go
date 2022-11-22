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
	srv := nvidia_inferenceserver.TritonClientService{ServerURL: "<Your Triton HTTP Host>:<Your Triton HTTP Port>"}
	httpErr := srv.InitTritonConnection(nil, nil)
	if httpErr != nil {
		panic(httpErr)
	}
	isReady, err := srv.CheckModelReady("<Your Model Name>", "<Your Model Version>", 1*time.Second, false)
	if err != nil {
		panic(err)
	}
	fmt.Println(isReady)
}

func TestTritonGRPCClientForCheckModelReady(t *testing.T) {
	srv := nvidia_inferenceserver.TritonClientService{ServerURL: "<Your Triton GRPC Host>:<Your Triton GRPC Port>"}
	grpcErr := srv.InitTritonConnection(nil, nil)
	if grpcErr != nil {
		panic(grpcErr)
	}
	isReady, err := srv.CheckModelReady("<Your Model Name>", "<Your Model Version>", 1*time.Second, true)
	if err != nil {
		panic(err)
	}
	fmt.Println(isReady)
}

func TestTritonHTTPClientInit(t *testing.T) {
	client := &fasthttp.Client{}
	trtClient := nvidia_inferenceserver.NewTritonClientWithOnlyHttp("<Your Triton HTTP Host>:<Your Triton HTTP Port>", client)
	isReady, err := trtClient.CheckServerReady(1*time.Second, false)
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
	isReady, err := trtClient.CheckServerReady(1*time.Second, false)
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
	isReady, err := trtClient.CheckServerReady(1*time.Second, false)
	if err != nil {
		panic(err)
	}
	fmt.Println(isReady)
}
