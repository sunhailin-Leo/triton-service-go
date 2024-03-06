package test

import (
	"reflect"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

func TestPadSlice(t *testing.T) {
	mySlice := []string{"apple", "orange", "banana"}

	padLength := 6
	paddingValue := "grape"

	paddedSlice := utils.PadSlice(mySlice, padLength, paddingValue)
	expected := []string{"apple", "orange", "banana", "grape", "grape", "grape"}
	if !reflect.DeepEqual(expected, paddedSlice) {
		t.Fatalf("assertion failed, unexpected: %v, expected: %v", paddedSlice, expected)
	}
}

func TestStringSliceTruncate(t *testing.T) {
	input1 := [][]string{
		{"a", "b", "c"},
		{"d", "e", "f"},
	}
	maxLen1 := 2
	expected1 := [][]string{
		{"a", "b", "c"},
		{"d", "e"},
	}

	result1 := utils.StringSliceTruncate(input1, maxLen1)
	if !reflect.DeepEqual(result1, expected1) {
		t.Errorf("Test case 1 failed. Expected %v, got %v", expected1, result1)
	}
}

func TestSliceTransposeFor3D(t *testing.T) {
	input1 := [][][]int{
		{{1, 2, 3}, {4, 5, 6}},
		{{7, 8, 9}, {10, 11, 12}},
	}
	expected1 := [][][]int{
		{{1, 2, 3}, {7, 8, 9}},
		{{4, 5, 6}, {10, 11, 12}},
	}
	result1 := utils.SliceTransposeFor3D(input1)
	if !reflect.DeepEqual(result1, expected1) {
		t.Errorf("Test case 1 failed. Expected %v, got %v", expected1, result1)
	}
}

func TestSliceTransposeFor2D(t *testing.T) {
	input1 := [][]int{
		{1, 2, 3},
		{4, 5, 6},
	}
	expected1 := [][]int{
		{1, 4},
		{2, 5},
		{3, 6},
	}
	result1 := utils.SliceTransposeFor2D(input1)
	if !reflect.DeepEqual(result1, expected1) {
		t.Errorf("Test case 1 failed. Expected %v, got %v", expected1, result1)
	}
}

func TestFlatten2DSlice(t *testing.T) {
	input1 := [][]int{{1, 2, 3}, {4, 5, 6}}
	expected1 := []int{1, 2, 3, 4, 5, 6}
	result1 := utils.Flatten2DSlice(input1)
	if !reflect.DeepEqual(result1, expected1) {
		t.Errorf("Test case 1 failed. Expected %v, got %v", expected1, result1)
	}
}

func TestGetMaxSubSliceLength(t *testing.T) {
	input1 := [][]int{{1, 2, 3}, {4, 5}}
	expected1 := 3
	result1 := utils.GetMaxSubSliceLength(input1)
	if !reflect.DeepEqual(result1, expected1) {
		t.Errorf("Test case 1 failed. Expected %v, got %v", expected1, result1)
	}
}

func TestSliceToInterfaceSlice(t *testing.T) {
	input1 := [][]int{{1, 2, 3}, {4, 5, 6}}
	expected1 := []interface{}{[]int{1, 2, 3}, []int{4, 5, 6}}
	result1 := utils.SliceToInterfaceSlice(input1)
	if !reflect.DeepEqual(result1, expected1) {
		t.Errorf("Test case 1 failed. Expected %v, got %v", expected1, result1)
	}
}

func TestGenerateRange(t *testing.T) {
	expected1 := []int{0, 1, 2, 3, 4}
	result1 := utils.GenerateRange[int](0, 5)
	if !reflect.DeepEqual(result1, expected1) {
		t.Errorf("Test case 1 failed. Expected %v, got %v", expected1, result1)
	}
}
