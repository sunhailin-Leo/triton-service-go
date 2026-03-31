package test

import (
	"fmt"
	"strings"
	"testing"

	daulet "github.com/daulet/tokenizers"
	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
)

// ==================== Correctness Comparison Tests ====================

// TestCompareChineseTokenization compares Chinese tokenization results between
// the current pure-Go WordPiece tokenizer and daulet/tokenizers (HuggingFace Rust binding).
func TestCompareChineseTokenization(t *testing.T) {
	// --- Current tokenizer (pure Go) ---
	goVocab, err := transformers.VocabFromFile("bert-chinese-vocab.txt")
	if err != nil {
		t.Fatalf("failed to load Go vocab: %v", err)
	}
	goTokenizer := transformers.NewWordPieceTokenizer(goVocab)

	// --- daulet/tokenizers (Rust FFI) ---
	rustTokenizer, err := daulet.FromFile("bert-chinese-tokenizer.json")
	if err != nil {
		t.Fatalf("failed to load Rust tokenizer: %v", err)
	}
	defer rustTokenizer.Close()

	testCases := []struct {
		name string
		text string
	}{
		{"pure_chinese", "广东省深圳市南山区腾讯大厦"},
		{"chinese_with_punctuation", "你好，世界！"},
		{"chinese_english_mixed", "深圳是一个beautiful的城市"},
		{"chinese_with_numbers", "2024年深圳GDP超过3万亿"},
		{"short_chinese", "你好"},
		{"single_char", "深"},
		{"long_chinese", "中华人民共和国是一个伟大的社会主义国家拥有悠久的历史和灿烂的文化"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Go tokenizer: Chinese mode
			goResult := goTokenizer.TokenizeChinese(tc.text)
			goTokens := make([]string, len(goResult))
			for i, tok := range goResult {
				goTokens[i] = tok.String
			}

			// Rust tokenizer: encode without special tokens
			rustIDs, rustTokens := rustTokenizer.Encode(tc.text, false)

			t.Logf("Input: %q", tc.text)
			t.Logf("Go tokens (%d):   %v", len(goTokens), goTokens)
			t.Logf("Rust tokens (%d): %v", len(rustTokens), rustTokens)
			t.Logf("Rust IDs:         %v", rustIDs)

			// Compare token count
			if len(goTokens) != len(rustTokens) {
				t.Logf("⚠️  Token count differs: Go=%d, Rust=%d", len(goTokens), len(rustTokens))
			}

			// Compare token strings (normalize ## prefix)
			matchCount := 0
			minLen := len(goTokens)
			if len(rustTokens) < minLen {
				minLen = len(rustTokens)
			}
			for i := 0; i < minLen; i++ {
				if goTokens[i] == rustTokens[i] {
					matchCount++
				}
			}
			matchRate := float64(matchCount) / float64(max(len(goTokens), len(rustTokens))) * 100
			t.Logf("✅ Match rate: %.1f%% (%d/%d)", matchRate, matchCount, max(len(goTokens), len(rustTokens)))
		})
	}
}

// TestCompareMultilingualTokenization compares non-Chinese tokenization results.
func TestCompareMultilingualTokenization(t *testing.T) {
	// --- Current tokenizer (pure Go) ---
	goVocab, err := transformers.VocabFromFile("bert-multilingual-vocab.txt")
	if err != nil {
		t.Fatalf("failed to load Go vocab: %v", err)
	}
	goTokenizer := transformers.NewWordPieceTokenizer(goVocab)

	// --- daulet/tokenizers (Rust FFI) ---
	rustTokenizer, err := daulet.FromFile("bert-multilingual-tokenizer.json")
	if err != nil {
		t.Fatalf("failed to load Rust tokenizer: %v", err)
	}
	defer rustTokenizer.Close()

	testCases := []struct {
		name string
		text string
	}{
		{"english_simple", "Hello World"},
		{"english_sentence", "The quick brown fox jumps over the lazy dog"},
		{"thai_text", "นครปฐม เมืองนครปฐม ถนนขาด เลขที่ 69 หมู่ 1 ซ. - - ถ. -"},
		{"mixed_punctuation", "Hello, World! How are you?"},
		{"numbers", "The price is $42.50"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Go tokenizer
			goResult := goTokenizer.Tokenize(tc.text)
			goTokens := make([]string, len(goResult))
			for i, tok := range goResult {
				goTokens[i] = tok.String
			}

			// Rust tokenizer: encode without special tokens
			rustIDs, rustTokens := rustTokenizer.Encode(tc.text, false)

			t.Logf("Input: %q", tc.text)
			t.Logf("Go tokens (%d):   %v", len(goTokens), goTokens)
			t.Logf("Rust tokens (%d): %v", len(rustTokens), rustTokens)
			t.Logf("Rust IDs:         %v", rustIDs)

			// Compare
			matchCount := 0
			minLen := len(goTokens)
			if len(rustTokens) < minLen {
				minLen = len(rustTokens)
			}
			for i := 0; i < minLen; i++ {
				if goTokens[i] == rustTokens[i] {
					matchCount++
				}
			}
			matchRate := float64(matchCount) / float64(max(len(goTokens), len(rustTokens))) * 100
			t.Logf("✅ Match rate: %.1f%% (%d/%d)", matchRate, matchCount, max(len(goTokens), len(rustTokens)))
		})
	}
}

// TestCompareTokenIDs compares token IDs between Go and Rust tokenizers for Chinese text.
func TestCompareTokenIDs(t *testing.T) {
	// --- Current tokenizer (pure Go) ---
	goVocab, err := transformers.VocabFromFile("bert-chinese-vocab.txt")
	if err != nil {
		t.Fatalf("failed to load Go vocab: %v", err)
	}
	goTokenizer := transformers.NewWordPieceTokenizer(goVocab)

	// --- daulet/tokenizers (Rust FFI) ---
	rustTokenizer, err := daulet.FromFile("bert-chinese-tokenizer.json")
	if err != nil {
		t.Fatalf("failed to load Rust tokenizer: %v", err)
	}
	defer rustTokenizer.Close()

	text := "广东省深圳市南山区腾讯大厦"

	// Go: get token IDs
	goResult := goTokenizer.TokenizeChinese(text)
	goTokenStrs := make([]string, len(goResult))
	for i, tok := range goResult {
		goTokenStrs[i] = tok.String
	}
	goIDs := goVocab.ConvertItems(goTokenStrs)

	// Rust: get token IDs
	rustIDs, rustTokens := rustTokenizer.Encode(text, false)

	t.Logf("Input: %q", text)
	t.Logf("Go tokens:  %v", goTokenStrs)
	t.Logf("Go IDs:     %v", goIDs)
	t.Logf("Rust tokens: %v", rustTokens)
	t.Logf("Rust IDs:    %v", rustIDs)

	// Compare IDs
	if len(goIDs) != len(rustIDs) {
		t.Logf("⚠️  ID count differs: Go=%d, Rust=%d", len(goIDs), len(rustIDs))
	}
	matchCount := 0
	minLen := len(goIDs)
	if len(rustIDs) < minLen {
		minLen = len(rustIDs)
	}
	for i := 0; i < minLen; i++ {
		if int32(rustIDs[i]) == int32(goIDs[i]) {
			matchCount++
		} else {
			t.Logf("  ID mismatch at %d: Go=%d, Rust=%d (Go=%q, Rust=%q)",
				i, goIDs[i], rustIDs[i], goTokenStrs[i], rustTokens[i])
		}
	}
	matchRate := float64(matchCount) / float64(max(len(goIDs), len(rustIDs))) * 100
	t.Logf("✅ Token ID match rate: %.1f%% (%d/%d)", matchRate, matchCount, max(len(goIDs), len(rustIDs)))
}

// TestCompareOffsets compares offset tracking between Go and Rust tokenizers.
func TestCompareOffsets(t *testing.T) {
	// --- Current tokenizer (pure Go) ---
	goVocab, err := transformers.VocabFromFile("bert-chinese-vocab.txt")
	if err != nil {
		t.Fatalf("failed to load Go vocab: %v", err)
	}
	goTokenizer := transformers.NewWordPieceTokenizer(goVocab)

	// --- daulet/tokenizers (Rust FFI) ---
	rustTokenizer, err := daulet.FromFile("bert-chinese-tokenizer.json")
	if err != nil {
		t.Fatalf("failed to load Rust tokenizer: %v", err)
	}
	defer rustTokenizer.Close()

	text := "广东省深圳市南山区腾讯大厦"

	// Go: get offsets
	goResult := goTokenizer.TokenizeChinese(text)

	// Rust: get offsets
	encoding := rustTokenizer.EncodeWithOptions(text, false,
		daulet.WithReturnTokens(),
		daulet.WithReturnOffsets(),
	)

	t.Logf("Input: %q", text)
	t.Logf("--- Go Offsets ---")
	for i, tok := range goResult {
		t.Logf("  [%d] %q  offset=(%d, %d)", i, tok.String, tok.Offsets.Start, tok.Offsets.End)
	}
	t.Logf("--- Rust Offsets ---")
	for i, tok := range encoding.Tokens {
		t.Logf("  [%d] %q  offset=(%d, %d)", i, tok, encoding.Offsets[i][0], encoding.Offsets[i][1])
	}

	// Note: Rust offsets are byte-based, Go offsets are rune-based
	// For pure ASCII/single-byte chars they match, for CJK they differ
	t.Logf("⚠️  Note: Rust offsets are byte-based, Go offsets are rune-based (CJK chars = 3 bytes)")
}

// TestCompareSpecialTokens compares special token handling.
func TestCompareSpecialTokens(t *testing.T) {
	// --- daulet/tokenizers (Rust FFI) ---
	rustTokenizer, err := daulet.FromFile("bert-chinese-tokenizer.json")
	if err != nil {
		t.Fatalf("failed to load Rust tokenizer: %v", err)
	}
	defer rustTokenizer.Close()

	text := "广东省深圳市"

	// Without special tokens
	idsNoSpecial, tokensNoSpecial := rustTokenizer.Encode(text, false)
	t.Logf("Without special tokens: %v %v", idsNoSpecial, tokensNoSpecial)

	// With special tokens ([CLS] ... [SEP])
	idsWithSpecial, tokensWithSpecial := rustTokenizer.Encode(text, true)
	t.Logf("With special tokens:    %v %v", idsWithSpecial, tokensWithSpecial)

	// Verify [CLS] and [SEP] are added
	if len(tokensWithSpecial) != len(tokensNoSpecial)+2 {
		t.Errorf("expected %d tokens with special, got %d", len(tokensNoSpecial)+2, len(tokensWithSpecial))
	}
	if tokensWithSpecial[0] != "[CLS]" {
		t.Errorf("expected first token to be [CLS], got %q", tokensWithSpecial[0])
	}
	if tokensWithSpecial[len(tokensWithSpecial)-1] != "[SEP]" {
		t.Errorf("expected last token to be [SEP], got %q", tokensWithSpecial[len(tokensWithSpecial)-1])
	}
}

// ==================== Performance Benchmark Tests ====================

// BenchmarkGoTokenizerChinese benchmarks the current pure-Go tokenizer for Chinese text.
func BenchmarkGoTokenizerChinese(b *testing.B) {
	goVocab, err := transformers.VocabFromFile("bert-chinese-vocab.txt")
	if err != nil {
		b.Fatalf("failed to load vocab: %v", err)
	}
	goTokenizer := transformers.NewWordPieceTokenizer(goVocab)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		goTokenizer.TokenizeChinese("广东省深圳市南山区腾讯大厦")
	}
	b.ReportAllocs()
}

// BenchmarkRustTokenizerChinese benchmarks daulet/tokenizers for Chinese text.
func BenchmarkRustTokenizerChinese(b *testing.B) {
	rustTokenizer, err := daulet.FromFile("bert-chinese-tokenizer.json")
	if err != nil {
		b.Fatalf("failed to load tokenizer: %v", err)
	}
	defer rustTokenizer.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rustTokenizer.Encode("广东省深圳市南山区腾讯大厦", false)
	}
	b.ReportAllocs()
}

// BenchmarkGoTokenizerMultilingual benchmarks the current pure-Go tokenizer for non-Chinese text.
func BenchmarkGoTokenizerMultilingual(b *testing.B) {
	goVocab, err := transformers.VocabFromFile("bert-multilingual-vocab.txt")
	if err != nil {
		b.Fatalf("failed to load vocab: %v", err)
	}
	goTokenizer := transformers.NewWordPieceTokenizer(goVocab)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		goTokenizer.Tokenize("นครปฐม เมืองนครปฐม ถนนขาด เลขที่ 69 หมู่ 1 ซ. - - ถ. -")
	}
	b.ReportAllocs()
}

// BenchmarkRustTokenizerMultilingual benchmarks daulet/tokenizers for non-Chinese text.
func BenchmarkRustTokenizerMultilingual(b *testing.B) {
	rustTokenizer, err := daulet.FromFile("bert-multilingual-tokenizer.json")
	if err != nil {
		b.Fatalf("failed to load tokenizer: %v", err)
	}
	defer rustTokenizer.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rustTokenizer.Encode("นครปฐม เมืองนครปฐม ถนนขาด เลขที่ 69 หมู่ 1 ซ. - - ถ. -", false)
	}
	b.ReportAllocs()
}

// BenchmarkGoTokenizerLongChinese benchmarks Go tokenizer with longer Chinese text.
func BenchmarkGoTokenizerLongChinese(b *testing.B) {
	goVocab, err := transformers.VocabFromFile("bert-chinese-vocab.txt")
	if err != nil {
		b.Fatalf("failed to load vocab: %v", err)
	}
	goTokenizer := transformers.NewWordPieceTokenizer(goVocab)
	longText := strings.Repeat("中华人民共和国是一个伟大的社会主义国家", 5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		goTokenizer.TokenizeChinese(longText)
	}
	b.ReportAllocs()
}

// BenchmarkRustTokenizerLongChinese benchmarks Rust tokenizer with longer Chinese text.
func BenchmarkRustTokenizerLongChinese(b *testing.B) {
	rustTokenizer, err := daulet.FromFile("bert-chinese-tokenizer.json")
	if err != nil {
		b.Fatalf("failed to load tokenizer: %v", err)
	}
	defer rustTokenizer.Close()
	longText := strings.Repeat("中华人民共和国是一个伟大的社会主义国家", 5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rustTokenizer.Encode(longText, false)
	}
	b.ReportAllocs()
}

// ==================== Impact Analysis ====================

// TestImpactAnalysis prints a summary of what would need to change if daulet/tokenizers is adopted.
func TestImpactAnalysis(t *testing.T) {
	report := `
╔══════════════════════════════════════════════════════════════════════╗
║              daulet/tokenizers 引入影响分析报告                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. 配置文件变更                                                     ║
║     - 当前: vocab.txt (逐行 token 文本)                              ║
║     - 引入后: tokenizer.json (HuggingFace 格式)                     ║
║     - 影响: 用户需要提供 tokenizer.json 而非 vocab.txt              ║
║                                                                      ║
║  2. 数据类型变更                                                     ║
║     - Token ID: int32 → uint32                                       ║
║     - Offsets: OffsetsType{Start,End int} → [2]uint (byte-based)    ║
║     - 影响: bert.go 中 getBertInputFeature 需要适配类型转换         ║
║                                                                      ║
║  3. 中文分词差异                                                     ║
║     - 当前: 自定义 TokenizeChinese (逐字分割)                       ║
║     - daulet: 依赖 tokenizer.json 中的 pre_tokenizer 配置          ║
║     - 影响: 需验证 bert-base-chinese 的 tokenizer.json 是否         ║
║             自带中文逐字分割能力                                     ║
║                                                                      ║
║  4. 构建依赖变更                                                     ║
║     - 当前: 纯 Go，零外部依赖，CGO_ENABLED=0 可编译                ║
║     - 引入后: 需要 CGO + 预编译 libtokenizers.a                     ║
║     - 影响: CI/CD 需要配置 CGO 环境，跨平台编译复杂度增加           ║
║                                                                      ║
║  5. 需要改造的文件                                                   ║
║     - models/transformers/bert.go: 替换 tokenizer 调用              ║
║     - models/transformers/bert_w2ner.go: 替换 tokenizer 调用        ║
║     - models/transformers/common.go: 更新类型定义                   ║
║     - models/base.go: 更新 BertVocab 类型                           ║
║     - go.mod: 添加 daulet/tokenizers 依赖                           ║
║                                                                      ║
║  6. 可选方案                                                         ║
║     - 方案A: 完全替换 → 移除当前 tokenizer，全部使用 daulet         ║
║     - 方案B: 双模式 → 保留当前 tokenizer 作为 fallback，            ║
║              新增 daulet 作为高性能选项                               ║
║     - 方案C: 不引入 → 保持纯 Go 实现，优化现有代码                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
`
	fmt.Println(report)
}
