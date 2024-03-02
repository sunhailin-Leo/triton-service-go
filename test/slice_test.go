package test

import (
	"reflect"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/utils"
)

func TestPad(t *testing.T) {
	mySlice := []string{"apple", "orange", "banana"}

	padLength := 6
	paddingValue := "grape"

	paddedSlice := utils.PadSlice(mySlice, padLength, paddingValue)
	expected := []string{"apple", "orange", "banana", "grape", "grape", "grape"}
	if !reflect.DeepEqual(expected, paddedSlice) {
		t.Fatalf("assertion failed, unexpected: %v, expected: %v", paddedSlice, expected)
	}
}
