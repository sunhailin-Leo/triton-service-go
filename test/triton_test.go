package test

import (
	"fmt"
	"testing"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/nvidia_inferenceserver"
)

func TestTritonHTTPClientForCheckModelReady(t *testing.T) {
	srv := nvidia_inferenceserver.TritonClientService{ServerURL: "<Your Triton HTTP Host>:<Your Triton HTTP Port>"}
	httpErr := srv.ConnectToTritonWithHTTP(nil)
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
	grpcErr := srv.ConnectToTritonWithGRPC()
	if grpcErr != nil {
		panic(grpcErr)
	}
	isReady, err := srv.CheckModelReady("<Your Model Name>", "<Your Model Version>", 1*time.Second, true)
	if err != nil {
		panic(err)
	}
	fmt.Println(isReady)
}
