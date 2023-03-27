package test

import (
	"log"
	"strings"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/models/bert"
)

func TestFullTokenizerNotChinese(_ *testing.T) {
	voc, vocabReadErr := bert.VocabFromFile("bert-multilingual-vocab.txt")
	if vocabReadErr != nil {
		panic(vocabReadErr)
	}
	tokenizer := bert.NewWordPieceTokenizer(voc)
	tokenResult := tokenizer.Tokenize("นครปฐม เมืองนครปฐม ถนนขาด เลขที่ 69 หมู่ 1 ซ. - - ถ. -")
	var tokenStrArray = make([]string, len(tokenResult))
	var tokenOffsetArray = make([]bert.OffsetsType, len(tokenResult))
	for i, token := range tokenResult {
		tokenStrArray[i] = token.String
		tokenOffsetArray[i] = token.Offsets
	}
	log.Println(len(tokenStrArray), tokenStrArray, tokenOffsetArray)
}

// BenchmarkFullTokenizerNotChinese-12    	   56792	     19936 ns/op	   13912 B/op	     277 allocs/op
func BenchmarkFullTokenizerNotChinese(b *testing.B) {
	voc, vocabReadErr := bert.VocabFromFile("bert-multilingual-vocab.txt")
	if vocabReadErr != nil {
		panic(vocabReadErr)
	}
	tokenizer := bert.NewWordPieceTokenizer(voc)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.Tokenize("นครปฐม เมืองนครปฐม ถนนขาด เลขที่ 69 หมู่ 1 ซ. - - ถ. -")
	}
	b.ReportAllocs()
}

func TestFullTokenizerChinese(_ *testing.T) {
	voc, vocabReadErr := bert.VocabFromFile("bert-chinese-vocab.txt")
	if vocabReadErr != nil {
		panic(vocabReadErr)
	}
	tokenizer := bert.NewWordPieceTokenizer(voc)
	tokenResult := tokenizer.TokenizeChinese(strings.ToLower("广东省深圳市南山区腾讯大厦"))
	log.Println("Final: ", tokenResult)
	var tokenStrArray = make([]string, len(tokenResult))
	for i, token := range tokenResult {
		tokenStrArray[i] = token.String
		log.Println(token.String, token.Offsets)
	}
	log.Println(len(tokenStrArray), tokenStrArray)
}

// BenchmarkFullTokenizerChinese-12    	  162031	      7488 ns/op	    3920 B/op	     102 allocs/op
func BenchmarkFullTokenizerChinese(b *testing.B) {
	voc, vocabReadErr := bert.VocabFromFile("bert-chinese-vocab.txt")
	if vocabReadErr != nil {
		panic(vocabReadErr)
	}
	tokenizer := bert.NewWordPieceTokenizer(voc)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.TokenizeChinese(strings.ToLower("广东省深圳市南山区腾讯大厦"))
	}
	b.ReportAllocs()
}
