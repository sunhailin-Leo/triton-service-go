package test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/models/bert"
)

func TestFullTokenizerNotChinese(t *testing.T) {
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
	fmt.Println(len(tokenStrArray), tokenStrArray, tokenOffsetArray)
}

func TestFullTokenizerChinese(t *testing.T) {
	voc, vocabReadErr := bert.VocabFromFile("../../static/bert-chinese-vocab.txt")
	//voc, vocabReadErr := VocabFromFile("../../static/bert-multilingual-vocab.txt")
	if vocabReadErr != nil {
		panic(vocabReadErr)
	}
	tokenizer := bert.NewWordPieceTokenizer(voc)
	tokenResult := tokenizer.TokenizeChinese(strings.ToLower("广东省深圳市南山区腾讯大厦"))
	fmt.Println("Final: ", tokenResult)
	var tokenStrArray = make([]string, len(tokenResult))
	for i, token := range tokenResult {
		tokenStrArray[i] = token.String
		fmt.Println(token.String, token.Offsets)
	}
	fmt.Println(len(tokenStrArray), tokenStrArray)
}
