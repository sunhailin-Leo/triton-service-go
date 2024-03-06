package test

import (
	"errors"
	"reflect"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

func TestNewVocabFromFile(t *testing.T) {
	vocabPath := "bert-chinese-vocab.txt"
	_, vocabReadErr := transformers.VocabFromFile(vocabPath)
	if vocabReadErr != nil {
		t.Fatalf("read vocab file failed: %v", vocabReadErr)
	}
}

func TestNewVocabFromSlice(t *testing.T) {
	tests := []struct {
		input     []string
		wantError error
	}{
		{[]string{}, utils.ErrEmptyVocab},
		{[]string{"A", "B", "C"}, nil},
		// Add more test cases as needed
	}

	for _, test := range tests {
		_, vocabReadErr := transformers.VocabFromSlice(test.input)
		if !errors.Is(vocabReadErr, test.wantError) {
			t.Fatalf("input: %v, expected error: %v, got: %v", test.input, test.wantError, vocabReadErr)
		}
	}
}

func TestVocabAdd(t *testing.T) {
	initToken := []string{"A", "B", "C"}
	tests := []struct {
		input   string
		wantLen int
	}{
		{"D", 4},
		// Add more test cases as needed
	}

	for _, test := range tests {
		vocab, vocabReadErr := transformers.VocabFromSlice(initToken)
		if vocabReadErr != nil {
			t.Fatalf("read vocab file failed: %v", vocabReadErr)
		}
		vocab.Add(test.input)
		if vocab.Size() != test.wantLen {
			t.Fatalf("input: %v, expected: %v, got: %v", test.input, test.wantLen, vocab.Size())
		}
	}
}

func TestVocabGetID(t *testing.T) {
	initToken := []string{"A", "B", "C"}
	tests := []struct {
		input  string
		wantID int32
	}{
		{"A", 0},
		{"B", 1},
		{"C", 2},
		{"D", -1},
		// Add more test cases as needed
	}

	vocab, vocabReadErr := transformers.VocabFromSlice(initToken)
	if vocabReadErr != nil {
		t.Fatalf("read vocab file failed: %v", vocabReadErr)
	}

	for _, test := range tests {
		result := int32(vocab.GetID(test.input))
		if result != test.wantID {
			t.Fatalf("input: %v, expected: %v, got: %v", test.input, test.wantID, result)
		}
	}
}

func TestVocabSize(t *testing.T) {
	tests := []struct {
		input    []string
		wantSize int
	}{
		{[]string{"D"}, 1},
		{[]string{"A", "B", "C"}, 3},
		// Add more test cases as needed
	}
	for _, test := range tests {
		vocab, vocabReadErr := transformers.VocabFromSlice(test.input)
		if vocabReadErr != nil {
			t.Fatalf("read vocab file failed: %v", vocabReadErr)
		}
		if vocab.Size() != test.wantSize {
			t.Fatalf("input: %v, expected: %v, got: %v", test.input, test.wantSize, vocab.Size())
		}
	}
}

func TestVocabConvertItems(t *testing.T) {
	tests := []struct {
		input     []string
		wantItems []transformers.ID
	}{
		{[]string{"A", "B", "C"}, []transformers.ID{0, 1, 2}},
		// Add more test cases as needed
	}

	for _, test := range tests {
		vocab, vocabReadErr := transformers.VocabFromSlice(test.input)
		if vocabReadErr != nil {
			t.Fatalf("read vocab file failed: %v", vocabReadErr)
		}
		result := vocab.ConvertItems(test.input)
		if !reflect.DeepEqual(result, test.wantItems) {
			t.Fatalf("input: %v, expected: %v, got: %v", test.input, test.wantItems, result)
		}
	}
}

func TestVocabIsInVocab(t *testing.T) {
	initToken := []string{"A", "B", "C"}
	tests := []struct {
		input    string
		wantResp bool
	}{
		{"A", true},
		{"D", false},
		// Add more test cases as needed
	}
	vocab, vocabReadErr := transformers.VocabFromSlice(initToken)
	if vocabReadErr != nil {
		t.Fatalf("read vocab file failed: %v", vocabReadErr)
	}
	for _, test := range tests {
		result := vocab.IsInVocab(test.input)
		if result != test.wantResp {
			t.Fatalf("input: %v, expected: %v, got: %v", test.input, test.wantResp, result)
		}
	}
}
