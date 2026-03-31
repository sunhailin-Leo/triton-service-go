package transformers_test

import (
	"strings"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
)

func TestGetStrings(t *testing.T) {
	tokens := []transformers.StringOffsetsPair{
		{String: "hello", Offsets: transformers.OffsetsType{Start: 0, End: 5}},
		{String: "world", Offsets: transformers.OffsetsType{Start: 6, End: 11}},
	}
	strings := transformers.GetStrings(tokens)
	if len(strings) != 2 {
		t.Fatalf("expected length 2, got %d", len(strings))
	}
	if strings[0] != "hello" {
		t.Errorf("expected 'hello', got %q", strings[0])
	}
	if strings[1] != "world" {
		t.Errorf("expected 'world', got %q", strings[1])
	}
}

func TestGetOffsets(t *testing.T) {
	tokens := []transformers.StringOffsetsPair{
		{String: "hello", Offsets: transformers.OffsetsType{Start: 0, End: 5}},
		{String: "world", Offsets: transformers.OffsetsType{Start: 6, End: 11}},
	}
	offsets := transformers.GetOffsets(tokens)
	if len(offsets) != 2 {
		t.Fatalf("expected length 2, got %d", len(offsets))
	}
	if offsets[0].Start != 0 || offsets[0].End != 5 {
		t.Errorf("expected {0, 5}, got {%d, %d}", offsets[0].Start, offsets[0].End)
	}
	if offsets[1].Start != 6 || offsets[1].End != 11 {
		t.Errorf("expected {6, 11}, got {%d, %d}", offsets[1].Start, offsets[1].End)
	}
}

func TestNewBaseTokenizer(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer()
	if tokenizer == nil {
		t.Fatal("expected non-nil tokenizer")
	}
}

func TestRegisterSpecialWords(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer(
		transformers.RegisterSpecialWords("[CLS]", "[SEP]"),
	)
	tokens := tokenizer.Tokenize("[CLS] hello [SEP]")
	if len(tokens) != 3 {
		t.Fatalf("expected 3 tokens, got %d", len(tokens))
	}
	if tokens[0].String != "[CLS]" {
		t.Errorf("expected first token to be '[CLS]', got %q", tokens[0].String)
	}
	if tokens[2].String != "[SEP]" {
		t.Errorf("expected third token to be '[SEP]', got %q", tokens[2].String)
	}
}

func TestWithLowerCase(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer(
		transformers.WithLowerCase(true),
	)
	// Note: BaseTokenizer.Tokenize does NOT apply lowercase - it only splits.
	// Lowercase is applied in TokenizeChinese path.
	tokens := tokenizer.Tokenize("HELLO")
	if len(tokens) != 1 {
		t.Fatalf("expected 1 token, got %d", len(tokens))
	}
	if tokens[0].String != "HELLO" {
		t.Errorf("expected 'HELLO', got %q", tokens[0].String)
	}
}

func TestBaseTokenizerTokenize_English(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer()
	tokens := tokenizer.Tokenize("hello world")
	if len(tokens) != 2 {
		t.Fatalf("expected 2 tokens, got %d", len(tokens))
	}
	if tokens[0].String != "hello" {
		t.Errorf("expected 'hello', got %q", tokens[0].String)
	}
	if tokens[1].String != "world" {
		t.Errorf("expected 'world', got %q", tokens[1].String)
	}
}

func TestBaseTokenizerTokenize_Punctuation(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer()
	tokens := tokenizer.Tokenize("hello, world!")
	if len(tokens) != 4 {
		t.Fatalf("expected 4 tokens, got %d", len(tokens))
	}
	if tokens[0].String != "hello" {
		t.Errorf("expected 'hello', got %q", tokens[0].String)
	}
	if tokens[1].String != "," {
		t.Errorf("expected ',', got %q", tokens[1].String)
	}
	if tokens[2].String != "world" {
		t.Errorf("expected 'world', got %q", tokens[2].String)
	}
	if tokens[3].String != "!" {
		t.Errorf("expected '!', got %q", tokens[3].String)
	}
}

func TestBaseTokenizerTokenize_SpecialWords(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer(
		transformers.RegisterSpecialWords("[CLS]"),
	)
	tokens := tokenizer.Tokenize("[CLS] hello")
	if len(tokens) != 2 {
		t.Fatalf("expected 2 tokens, got %d", len(tokens))
	}
	if tokens[0].String != "[CLS]" {
		t.Errorf("expected '[CLS]', got %q", tokens[0].String)
	}
}

func TestBaseTokenizerTokenizeChinese(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer()
	tokens := tokenizer.TokenizeChinese("你好世界")
	// Chinese characters are split individually by TokenizeChinese
	if len(tokens) != 4 {
		t.Fatalf("expected 4 tokens, got %d", len(tokens))
	}
	expected := []string{"你", "好", "世", "界"}
	for i, exp := range expected {
		if tokens[i].String != exp {
			t.Errorf("token[%d]: expected %q, got %q", i, exp, tokens[i].String)
		}
	}
}

func TestBaseTokenizerTokenizeChineseCharMode(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer()
	tokens := tokenizer.TokenizeChineseCharMode("你好世界")
	// Chinese characters are split individually by TokenizeChineseCharMode
	if len(tokens) != 4 {
		t.Fatalf("expected 4 tokens, got %d", len(tokens))
	}
	expected := []string{"你", "好", "世", "界"}
	for i, exp := range expected {
		if tokens[i].String != exp {
			t.Errorf("token[%d]: expected %q, got %q", i, exp, tokens[i].String)
		}
	}
}

func TestNewWordPieceTokenizer(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	if tokenizer == nil {
		t.Fatal("expected non-nil tokenizer")
	}
}

func TestWordPieceTokenizerSetDoLowerCase(t *testing.T) {
	vocab, err := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	if err != nil {
		t.Fatalf("failed to load vocab: %v", err)
	}
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	tokenizer.SetDoLowerCase(true)
	// "hello" is in the chinese vocab
	tokens := tokenizer.Tokenize("HELLO")
	if len(tokens) < 1 {
		t.Fatalf("expected at least 1 token, got %d", len(tokens))
	}
	// With doLowerCase=true, "HELLO" becomes "hello" which should be found in vocab
	if tokens[0].String != "hello" {
		t.Errorf("expected 'hello', got %q", tokens[0].String)
	}
}

func TestWordPieceTokenizerTokenize_English(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	tokens := tokenizer.Tokenize("hello")
	if len(tokens) != 1 {
		t.Fatalf("expected 1 token, got %d", len(tokens))
	}
	if tokens[0].String != "hello" {
		t.Errorf("expected 'hello', got %q", tokens[0].String)
	}
}

func TestWordPieceTokenizerTokenizeChinese(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	tokens := tokenizer.TokenizeChinese("你好")
	if len(tokens) != 2 {
		t.Fatalf("expected 2 tokens, got %d", len(tokens))
	}
}

func TestWordPieceTokenizerTokenizeChineseCharMode(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	tokens := tokenizer.TokenizeChineseCharMode("你好")
	if len(tokens) != 2 {
		t.Fatalf("expected 2 tokens, got %d", len(tokens))
	}
}

func TestWordPieceTokenizerWordPieceTokenize_TooLong(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	longToken := transformers.StringOffsetsPair{
		String:  string(make([]rune, 201)),
		Offsets: transformers.OffsetsType{Start: 0, End: 201},
	}
	tokens := tokenizer.WordPieceTokenize([]transformers.StringOffsetsPair{longToken})
	if len(tokens) != 1 {
		t.Fatalf("expected 1 token, got %d", len(tokens))
	}
	if tokens[0].String != "[UNK]" {
		t.Errorf("expected '[UNK]', got %q", tokens[0].String)
	}
}

func TestWordPieceTokenizerWordPieceTokenize_Subword(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	token := transformers.StringOffsetsPair{
		String:  "unhappiness",
		Offsets: transformers.OffsetsType{Start: 0, End: 11},
	}
	tokens := tokenizer.WordPieceTokenize([]transformers.StringOffsetsPair{token})
	if len(tokens) == 0 {
		t.Fatal("expected at least 1 token")
	}
}

func TestIsDefaultSpecial(t *testing.T) {
	tests := []struct {
		name string
		word string
		want bool
	}{
		{"[CLS]", "[CLS]", true},
		{"[SEP]", "[SEP]", true},
		{"[UNK]", "[UNK]", true},
		{"[MASK]", "[MASK]", true},
		{"normal word", "hello", false},
		{"empty", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := transformers.IsDefaultSpecial(tt.word); got != tt.want {
				t.Errorf("IsDefaultSpecial() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGroupPieces(t *testing.T) {
	tokens := []transformers.StringOffsetsPair{
		{String: "hello", Offsets: transformers.OffsetsType{Start: 0, End: 5}},
		{String: "##world", Offsets: transformers.OffsetsType{Start: 5, End: 10}},
		{String: "test", Offsets: transformers.OffsetsType{Start: 11, End: 15}},
		{String: "##ing", Offsets: transformers.OffsetsType{Start: 15, End: 19}},
	}
	groups := transformers.GroupPieces(tokens)
	if len(groups) != 2 {
		t.Fatalf("expected 2 groups, got %d", len(groups))
	}
	if groups[0].Start != 0 || groups[0].End != 1 {
		t.Errorf("expected first group {0, 1}, got {%d, %d}", groups[0].Start, groups[0].End)
	}
	if groups[1].Start != 2 || groups[1].End != 3 {
		t.Errorf("expected second group {2, 3}, got {%d, %d}", groups[1].Start, groups[1].End)
	}
}

func TestMakeOffsetPairsFromGroups(t *testing.T) {
	text := "hello"
	tokens := []transformers.StringOffsetsPair{
		{String: "he", Offsets: transformers.OffsetsType{Start: 0, End: 2}},
		{String: "##llo", Offsets: transformers.OffsetsType{Start: 2, End: 5}},
	}
	groups := []transformers.TokensRange{
		{Start: 0, End: 1},
	}
	pairs := transformers.MakeOffsetPairsFromGroups(text, tokens, groups)
	if len(pairs) != 1 {
		t.Fatalf("expected 1 pair, got %d", len(pairs))
	}
	if pairs[0].String != "hello" {
		t.Errorf("expected 'hello', got %q", pairs[0].String)
	}
	if pairs[0].Offsets.Start != 0 || pairs[0].Offsets.End != 5 {
		t.Errorf("expected offsets {0, 5}, got {%d, %d}", pairs[0].Offsets.Start, pairs[0].Offsets.End)
	}
}

func BenchmarkBaseTokenize(b *testing.B) {
	tokenizer := transformers.NewBaseTokenizer()
	text := "The quick brown fox jumps over the lazy dog. This is a test sentence for benchmarking."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tokenizer.Tokenize(text)
	}
}

func BenchmarkBaseTokenizeChinese(b *testing.B) {
	tokenizer := transformers.NewBaseTokenizer()
	text := "这是一个测试句子，用于基准测试中文分词的性能。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tokenizer.TokenizeChinese(text)
	}
}

func BenchmarkWordPieceTokenize_Chinese(b *testing.B) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	text := "这是一个测试句子，用于基准测试中文分词的性能。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tokenizer.TokenizeChinese(text)
	}
}

func BenchmarkWordPieceTokenize_Multilingual(b *testing.B) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-multilingual-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	text := "This is a test sentence for benchmarking."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tokenizer.Tokenize(text)
	}
}

func BenchmarkWordPieceTokenize_ChineseCharMode(b *testing.B) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	text := "这是一个测试句子，用于基准测试中文分词的性能。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tokenizer.TokenizeChineseCharMode(text)
	}
}

func BenchmarkGetStrings(b *testing.B) {
	tokens := make([]transformers.StringOffsetsPair, 100)
	for i := range tokens {
		tokens[i] = transformers.StringOffsetsPair{
			String:  "test",
			Offsets: transformers.OffsetsType{Start: i * 5, End: (i + 1) * 5},
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = transformers.GetStrings(tokens)
	}
}

func BenchmarkGroupPieces(b *testing.B) {
	tokens := make([]transformers.StringOffsetsPair, 100)
	for i := range tokens {
		if i%2 == 0 {
			tokens[i] = transformers.StringOffsetsPair{
				String:  "token",
				Offsets: transformers.OffsetsType{Start: i * 5, End: (i + 1) * 5},
			}
		} else {
			tokens[i] = transformers.StringOffsetsPair{
				String:  "##sub",
				Offsets: transformers.OffsetsType{Start: i * 5, End: (i + 1) * 5},
			}
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = transformers.GroupPieces(tokens)
	}
}

// TestWordPieceTokenizer_WordPieceTokenize_Empty tests empty token input
func TestWordPieceTokenizer_WordPieceTokenize_Empty(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)
	tokens := tokenizer.WordPieceTokenize([]transformers.StringOffsetsPair{})
	if len(tokens) != 0 {
		t.Errorf("expected 0 tokens for empty input, got %d", len(tokens))
	}
}

// TestWordPieceTokenizer_WordPieceTokenize_SpecialTokens tests special tokens handling
func TestWordPieceTokenizer_WordPieceTokenize_SpecialTokens(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)

	tests := []struct {
		name  string
		token string
	}{
		{"[CLS]", "[CLS]"},
		{"[SEP]", "[SEP]"},
		{"[UNK]", "[UNK]"},
		{"[MASK]", "[MASK]"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			token := transformers.StringOffsetsPair{
				String:  tt.token,
				Offsets: transformers.OffsetsType{Start: 0, End: len(tt.token)},
			}
			result := tokenizer.WordPieceTokenize([]transformers.StringOffsetsPair{token})
			if len(result) != 1 {
				t.Errorf("expected 1 token, got %d", len(result))
			}
			if result[0].String != tt.token {
				t.Errorf("expected %q, got %q", tt.token, result[0].String)
			}
		})
	}
}

// TestWordPieceTokenizer_WordPieceTokenize_SubwordPrefix tests subword prefix handling
func TestWordPieceTokenizer_WordPieceTokenize_SubwordPrefix(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)

	// Test a word that might be split into subwords
	token := transformers.StringOffsetsPair{
		String:  "unhappiness",
		Offsets: transformers.OffsetsType{Start: 0, End: 11},
	}
	result := tokenizer.WordPieceTokenize([]transformers.StringOffsetsPair{token})

	// Should have at least one token
	if len(result) == 0 {
		t.Error("expected at least 1 token")
	}

	// If multiple tokens, subsequent ones should have ## prefix
	for i := 1; i < len(result); i++ {
		if !strings.HasPrefix(result[i].String, "##") {
			t.Errorf("expected subword prefix at index %d, got %q", i, result[i].String)
		}
	}
}

// TestWordPieceTokenizer_WordPieceTokenize_OffsetsPreserved tests that offsets are preserved
func TestWordPieceTokenizer_WordPieceTokenize_OffsetsPreserved(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)

	token := transformers.StringOffsetsPair{
		String:  "hello",
		Offsets: transformers.OffsetsType{Start: 10, End: 15},
	}
	result := tokenizer.WordPieceTokenize([]transformers.StringOffsetsPair{token})

	if len(result) == 0 {
		t.Fatal("expected at least 1 token")
	}

	// First token should start at original start offset
	if result[0].Offsets.Start != 10 {
		t.Errorf("expected start offset 10, got %d", result[0].Offsets.Start)
	}

	// Last token should end at original end offset
	lastToken := result[len(result)-1]
	if lastToken.Offsets.End != 15 {
		t.Errorf("expected end offset 15, got %d", lastToken.Offsets.End)
	}
}

// TestGroupPieces_Empty tests empty tokens
func TestGroupPieces_Empty(t *testing.T) {
	groups := transformers.GroupPieces([]transformers.StringOffsetsPair{})
	if len(groups) != 0 {
		t.Errorf("expected 0 groups, got %d", len(groups))
	}
}

// TestGroupPieces_SingleToken tests single token
func TestGroupPieces_SingleToken(t *testing.T) {
	tokens := []transformers.StringOffsetsPair{
		{String: "hello", Offsets: transformers.OffsetsType{Start: 0, End: 5}},
	}
	groups := transformers.GroupPieces(tokens)
	if len(groups) != 1 {
		t.Errorf("expected 1 group, got %d", len(groups))
	}
	if groups[0].Start != 0 || groups[0].End != 0 {
		t.Errorf("expected {0, 0}, got {%d, %d}", groups[0].Start, groups[0].End)
	}
}

// TestGroupPieces_MultipleSubwords tests multiple subwords
func TestGroupPieces_MultipleSubwords(t *testing.T) {
	tokens := []transformers.StringOffsetsPair{
		{String: "un", Offsets: transformers.OffsetsType{Start: 0, End: 2}},
		{String: "##known", Offsets: transformers.OffsetsType{Start: 2, End: 7}},
		{String: "word", Offsets: transformers.OffsetsType{Start: 8, End: 12}},
		{String: "##s", Offsets: transformers.OffsetsType{Start: 12, End: 13}},
	}
	groups := transformers.GroupPieces(tokens)
	if len(groups) != 2 {
		t.Errorf("expected 2 groups, got %d", len(groups))
	}
	if groups[0].Start != 0 || groups[0].End != 1 {
		t.Errorf("expected first group {0, 1}, got {%d, %d}", groups[0].Start, groups[0].End)
	}
	if groups[1].Start != 2 || groups[1].End != 3 {
		t.Errorf("expected second group {2, 3}, got {%d, %d}", groups[1].Start, groups[1].End)
	}
}

// TestMakeOffsetPairsFromGroups_Empty tests empty groups
func TestMakeOffsetPairsFromGroups_Empty(t *testing.T) {
	result := transformers.MakeOffsetPairsFromGroups("", []transformers.StringOffsetsPair{}, []transformers.TokensRange{})
	if len(result) != 0 {
		t.Errorf("expected 0 pairs, got %d", len(result))
	}
}

// TestBaseTokenizer_SplitOn tests splitOn method
func TestBaseTokenizer_SplitOn(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer()

	// Use reflection or test indirectly through Tokenize
	tokens := tokenizer.Tokenize("hello world")
	if len(tokens) != 2 {
		t.Errorf("expected 2 tokens, got %d", len(tokens))
	}
	if tokens[0].String != "hello" {
		t.Errorf("expected 'hello', got %q", tokens[0].String)
	}
	if tokens[1].String != "world" {
		t.Errorf("expected 'world', got %q", tokens[1].String)
	}
}

// TestBaseTokenizer_SplitOnChinese tests splitOnChinese method
func TestBaseTokenizer_SplitOnChinese(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer()

	// Test indirectly through TokenizeChinese
	tokens := tokenizer.TokenizeChinese("你好世界")
	if len(tokens) != 4 {
		t.Errorf("expected 4 tokens, got %d", len(tokens))
	}
}

// TestBaseTokenizer_WithLowerCase tests lowercase option
func TestBaseTokenizer_WithLowerCase(t *testing.T) {
	tokenizer := transformers.NewBaseTokenizer(transformers.WithLowerCase(true))

	// BaseTokenizer doesn't apply lowercase in Tokenize, but does in TokenizeChinese
	tokens := tokenizer.TokenizeChinese("HELLO")
	if len(tokens) == 0 {
		t.Fatal("expected at least 1 token")
	}
	// Should be lowercased
	if tokens[0].String != "hello" {
		t.Errorf("expected 'hello', got %q", tokens[0].String)
	}
}

// TestWordPieceTokenizer_SetDoLowerCase_Chaining tests method chaining
func TestWordPieceTokenizer_SetDoLowerCase_Chaining(t *testing.T) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokenizer := transformers.NewWordPieceTokenizer(vocab)

	// Test that SetDoLowerCase returns the tokenizer for chaining
	result := tokenizer.SetDoLowerCase(true)
	if result != tokenizer {
		t.Error("expected SetDoLowerCase to return the same tokenizer instance")
	}
	// Verify lowercase is applied by checking tokenization result
	tokens := tokenizer.Tokenize("HELLO")
	if len(tokens) == 0 {
		t.Fatal("expected at least 1 token")
	}
	// With doLowerCase=true, "HELLO" should become "hello"
	if tokens[0].String != "hello" {
		t.Errorf("expected 'hello', got %q", tokens[0].String)
	}
}
