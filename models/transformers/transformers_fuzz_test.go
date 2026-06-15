package transformers_test

import (
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

// FuzzTokenize fuzzes the BaseTokenizer.Tokenize function.
func FuzzTokenize(f *testing.F) {
	seeds := []string{
		"hello world",
		"Hello, World!",
		"测试test123",
		"",
		"  spaces  ",
		"special!@#$%",
		"a",
		"  \t\n  ",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	tokenizer := transformers.NewBaseTokenizer()
	f.Fuzz(func(t *testing.T, text string) {
		result := tokenizer.Tokenize(text)
		// Verify that the resulting tokens are non-empty (except for empty input)
		for _, pair := range result {
			if pair.String == "" {
				t.Errorf("Tokenize() returned empty token at offset %v", pair.Offsets)
			}
		}
	})
}

// FuzzTokenizeChinese fuzzes the BaseTokenizer.TokenizeChinese function.
func FuzzTokenizeChinese(f *testing.F) {
	seeds := []string{
		"中文测试",
		"hello中文world",
		"测试test123",
		"",
		"  中文  ",
		"中，文",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	tokenizer := transformers.NewBaseTokenizer(transformers.WithLowerCase(true))
	f.Fuzz(func(t *testing.T, text string) {
		result := tokenizer.TokenizeChinese(text)
		_ = result
	})
}

// FuzzTokenizeChineseCharMode fuzzes the BaseTokenizer.TokenizeChineseCharMode function.
func FuzzTokenizeChineseCharMode(f *testing.F) {
	seeds := []string{
		"中文测试",
		"hello中文world",
		"123中文456",
		"",
		"中，文",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	tokenizer := transformers.NewBaseTokenizer(transformers.WithLowerCase(false))
	f.Fuzz(func(t *testing.T, text string) {
		result := tokenizer.TokenizeChineseCharMode(text)
		_ = result
	})
}

// FuzzWordPieceTokenize fuzzes the WordPieceTokenizer.WordPieceTokenize function.
func FuzzWordPieceTokenize(f *testing.F) {
	seeds := []string{
		"hello",
		"running",
		"unhappiness",
		"",
		"the",
		"a",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	vocab, err := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	if err != nil {
		f.Fatalf("Failed to load vocab: %v", err)
	}
	tokenizer := transformers.NewWordPieceTokenizer(&vocab)
	f.Fuzz(func(t *testing.T, text string) {
		baseTokens := tokenizer.Tokenize(text)
		result := tokenizer.WordPieceTokenize(baseTokens)
		_ = result
	})
}

// FuzzPadChinese fuzzes utils.PadChinese via tokenizer.
func FuzzPadChineseDirect(f *testing.F) {
	seeds := []string{
		"中文测试",
		"hello world",
		"中文English混合",
		"",
		"中123文",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		result := utils.PadChinese(text)
		// Result should be consistent - running twice should give same result
		result2 := utils.PadChinese(text)
		if result != result2 {
			t.Errorf("PadChinese() not deterministic: %q != %q", result, result2)
		}
	})
}

// FuzzCleanAndPadChineseWithWhiteSpace fuzzes the CleanAndPadChineseWithWhiteSpace function.
func FuzzCleanAndPadChineseWithWhiteSpace(f *testing.F) {
	seeds := []string{
		"中文测试",
		"hello world",
		"中文English混合",
		"",
		"中\x00文",
		"hello\tworld",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		result := utils.CleanAndPadChineseWithWhiteSpace(text)
		_ = result
	})
}

// FuzzClean fuzzes the Clean function.
func FuzzCleanDirect(f *testing.F) {
	seeds := []string{
		"hello world",
		"hello\x00world",
		"hello\t\nworld",
		"",
		"\x00\x01\x07",
		"正常文本",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		result := utils.Clean(text)
		// Clean should never return a string with null or replacement chars
		for _, r := range result {
			if r == '\x00' || r == '\ufffd' {
				t.Errorf("Clean() result contains invalid rune: %q", r)
			}
		}
	})
}

// FuzzSplitPunctuation fuzzes the SplitPunctuation function.
func FuzzSplitPunctuationDirect(f *testing.F) {
	seeds := []string{
		"hello,world",
		"hello",
		"",
		",!",
		"中文，测试",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		result := utils.SplitPunctuation(text)
		_ = result
	})
}

// FuzzVocabLongestSubstring fuzzes Dict.LongestSubstring.
func FuzzVocabLongestSubstring(f *testing.F) {
	seeds := []string{
		"hello",
		"world",
		"的",
		"test",
		"",
		"abcdefg",
		"中文字符",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	vocab, err := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	if err != nil {
		f.Fatalf("Failed to load vocab: %v", err)
	}
	f.Fuzz(func(t *testing.T, token string) {
		result := vocab.LongestSubstring(token)
		// LongestSubstring should always return a substring of token (or empty)
		if result != "" && len(result) > len(token) {
			t.Errorf("LongestSubstring(%q) = %q, longer than input", token, result)
		}
		if result != "" && !isSubstring(result, token) {
			t.Errorf("LongestSubstring(%q) = %q, not a substring", token, result)
		}
	})
}

func isSubstring(sub, s string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
