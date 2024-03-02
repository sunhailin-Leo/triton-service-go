package test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/models/transformers"
)

func TestFullTokenizerNotChinese(t *testing.T) {
	voc, vocabReadErr := transformers.VocabFromFile("bert-multilingual-vocab.txt")
	if vocabReadErr != nil {
		panic(vocabReadErr)
	}
	tokenizer := transformers.NewWordPieceTokenizer(voc)
	tokenResult := tokenizer.Tokenize("นครปฐม เมืองนครปฐม ถนนขาด เลขที่ 69 หมู่ 1 ซ. - - ถ. -")
	tokenStrSlice := make([]string, len(tokenResult))
	tokenOffsetSlice := make([]transformers.OffsetsType, len(tokenResult))
	for i, token := range tokenResult {
		tokenStrSlice[i] = token.String
		tokenOffsetSlice[i] = token.Offsets
	}

	// Test
	expectedLen := 34
	if len(tokenStrSlice) != expectedLen {
		t.Errorf("Expected '%d', but got '%d'", len(tokenStrSlice), expectedLen)
	}

	expectedTokenSlice := []string{
		"น", "##คร", "##ป", "##ฐ", "##ม", "เ", "##มือง", "##น", "##คร", "##ป", "##ฐ", "##ม",
		"ถ", "##น", "##น", "##ข", "##า", "##ด", "เ", "##ล", "##ข", "##ที่", "69", "ห", "##ม",
		"##ู่", "1", "ซ", ".", "-", "-", "ถ", ".", "-",
	}
	if !reflect.DeepEqual(tokenStrSlice, expectedTokenSlice) {
		t.Errorf("Expected '%v', but got '%v'", expectedTokenSlice, tokenStrSlice)
	}

	expectedTokenOffsetSlice := []transformers.OffsetsType{
		{Start: 0, End: 1},
		{Start: 1, End: 3},
		{Start: 3, End: 4},
		{Start: 4, End: 5},
		{Start: 5, End: 6},
		{Start: 7, End: 8},
		{Start: 8, End: 12},
		{Start: 12, End: 13},
		{Start: 13, End: 15},
		{Start: 15, End: 16},
		{Start: 16, End: 17},
		{Start: 17, End: 18},
		{Start: 19, End: 20},
		{Start: 20, End: 21},
		{Start: 21, End: 22},
		{Start: 22, End: 23},
		{Start: 23, End: 24},
		{Start: 24, End: 25},
		{Start: 26, End: 27},
		{Start: 27, End: 28},
		{Start: 28, End: 29},
		{Start: 29, End: 32},
		{Start: 33, End: 35},
		{Start: 36, End: 37},
		{Start: 37, End: 38},
		{Start: 38, End: 40},
		{Start: 41, End: 42},
		{Start: 43, End: 44},
		{Start: 44, End: 45},
		{Start: 46, End: 47},
		{Start: 48, End: 49},
		{Start: 50, End: 51},
		{Start: 51, End: 52},
		{Start: 53, End: 54},
	}
	if !reflect.DeepEqual(tokenOffsetSlice, expectedTokenOffsetSlice) {
		t.Errorf("Expected '%v', but got '%v'", expectedTokenOffsetSlice, tokenOffsetSlice)
	}
}

// BenchmarkFullTokenizerNotChinese-12    	   56792	     19936 ns/op	   13912 B/op	     277 allocs/op
func BenchmarkFullTokenizerNotChinese(b *testing.B) {
	voc, vocabReadErr := transformers.VocabFromFile("bert-multilingual-vocab.txt")
	if vocabReadErr != nil {
		panic(vocabReadErr)
	}
	tokenizer := transformers.NewWordPieceTokenizer(voc)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.Tokenize("นครปฐม เมืองนครปฐม ถนนขาด เลขที่ 69 หมู่ 1 ซ. - - ถ. -")
	}
	b.ReportAllocs()
}

func TestFullTokenizerChinese(t *testing.T) {
	voc, vocabReadErr := transformers.VocabFromFile("bert-chinese-vocab.txt")
	if vocabReadErr != nil {
		panic(vocabReadErr)
	}
	tokenizer := transformers.NewWordPieceTokenizer(voc)
	tokenResult := tokenizer.TokenizeChinese(strings.ToLower("广东省深圳市南山区腾讯大厦"))
	tokenStrSlice := make([]string, len(tokenResult))
	tokenOffsetSlice := make([]transformers.OffsetsType, len(tokenResult))
	for i, token := range tokenResult {
		tokenStrSlice[i] = token.String
		tokenOffsetSlice[i] = token.Offsets
	}

	// Test
	expectedLen := 13
	if len(tokenStrSlice) != expectedLen {
		t.Errorf("Expected '%d', but got '%d'", len(tokenStrSlice), expectedLen)
	}

	expectedTokenSlice := []string{"广", "东", "省", "深", "圳", "市", "南", "山", "区", "腾", "讯", "大", "厦"}
	if !reflect.DeepEqual(tokenStrSlice, expectedTokenSlice) {
		t.Errorf("Expected '%v', but got '%v'", expectedTokenSlice, tokenStrSlice)
	}

	expectedTokenOffsetSlice := []transformers.OffsetsType{
		{Start: 0, End: 1},
		{Start: 1, End: 2},
		{Start: 2, End: 3},
		{Start: 3, End: 4},
		{Start: 4, End: 5},
		{Start: 5, End: 6},
		{Start: 6, End: 7},
		{Start: 7, End: 8},
		{Start: 8, End: 9},
		{Start: 9, End: 10},
		{Start: 10, End: 11},
		{Start: 11, End: 12},
		{Start: 12, End: 13},
	}
	if !reflect.DeepEqual(tokenOffsetSlice, expectedTokenOffsetSlice) {
		t.Errorf("Expected '%v', but got '%v'", expectedTokenOffsetSlice, tokenOffsetSlice)
	}
}

// BenchmarkFullTokenizerChinese-12    	  162031	      7488 ns/op	    3920 B/op	     102 allocs/op
func BenchmarkFullTokenizerChinese(b *testing.B) {
	voc, vocabReadErr := transformers.VocabFromFile("bert-chinese-vocab.txt")
	if vocabReadErr != nil {
		panic(vocabReadErr)
	}
	tokenizer := transformers.NewWordPieceTokenizer(voc)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.TokenizeChinese(strings.ToLower("广东省深圳市南山区腾讯大厦"))
	}
	b.ReportAllocs()
}
