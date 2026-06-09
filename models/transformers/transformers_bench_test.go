package transformers_test

import (
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
)

// --- Tokenizer Benchmarks ---

func BenchmarkBaseTokenizer_Tokenize(b *testing.B) {
	tok := transformers.NewBaseTokenizer()
	text := "Hello, world! This is a benchmark test for the tokenizer."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Tokenize(text)
	}
}

func BenchmarkBaseTokenizer_TokenizeChinese(b *testing.B) {
	tok := transformers.NewBaseTokenizer(transformers.WithLowerCase(true))
	text := "这是一个中文分词的基准测试。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.TokenizeChinese(text)
	}
}

func BenchmarkBaseTokenizer_TokenizeChineseCharMode(b *testing.B) {
	tok := transformers.NewBaseTokenizer(transformers.WithLowerCase(false))
	text := "这是一个中文NER任务的基准测试。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.TokenizeChineseCharMode(text)
	}
}

func BenchmarkBaseTokenizer_Tokenize_LongText(b *testing.B) {
	tok := transformers.NewBaseTokenizer()
	text := "The quick brown fox jumps over the lazy dog. This is a much longer text that contains multiple sentences and various punctuation marks, such as: commas, periods, exclamation marks! And even some question marks? We want to test how the tokenizer performs on longer inputs with more tokens to process."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Tokenize(text)
	}
}

func BenchmarkBaseTokenizer_TokenizeChinese_LongText(b *testing.B) {
	tok := transformers.NewBaseTokenizer(transformers.WithLowerCase(true))
	text := "自然语言处理是人工智能领域中的一个重要方向。它涉及计算机与人类语言之间的交互，包括语音识别、自然语言理解、自然语言生成等多个子领域。近年来，随着深度学习技术的发展，自然语言处理取得了显著的进展。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.TokenizeChinese(text)
	}
}

func BenchmarkBaseTokenizer_Tokenize_MixedText(b *testing.B) {
	tok := transformers.NewBaseTokenizer(transformers.WithLowerCase(true))
	text := "BERT模型在NLP任务中表现优异，特别是中文BERT处理中文文本效果很好。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.TokenizeChinese(text)
	}
}

// --- WordPiece Tokenizer Benchmarks ---

func BenchmarkWordPieceTokenizer_Tokenize(b *testing.B) {
	vocab, err := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	if err != nil {
		b.Fatalf("Failed to load vocab: %v", err)
	}
	tok := transformers.NewWordPieceTokenizer(vocab)
	tok.SetDoLowerCase(true)
	text := "Hello, world! This is a benchmark test for the wordpiece tokenizer."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Tokenize(text)
	}
}

func BenchmarkWordPieceTokenizer_TokenizeChinese(b *testing.B) {
	vocab, err := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	if err != nil {
		b.Fatalf("Failed to load vocab: %v", err)
	}
	tok := transformers.NewWordPieceTokenizer(vocab)
	text := "自然语言处理是人工智能领域中的重要方向。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.TokenizeChinese(text)
	}
}

func BenchmarkWordPieceTokenizer_TokenizeChineseCharMode(b *testing.B) {
	vocab, err := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	if err != nil {
		b.Fatalf("Failed to load vocab: %v", err)
	}
	tok := transformers.NewWordPieceTokenizer(vocab)
	text := "自然语言处理是人工智能领域中的重要方向。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.TokenizeChineseCharMode(text)
	}
}

func BenchmarkWordPieceTokenizer_WordPieceTokenize(b *testing.B) {
	vocab, err := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	if err != nil {
		b.Fatalf("Failed to load vocab: %v", err)
	}
	tok := transformers.NewWordPieceTokenizer(vocab)
	text := "unaffordable"
	tokens := tok.Tokenize(text)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.WordPieceTokenize(tokens)
	}
}

// --- Vocab Benchmarks ---

func BenchmarkVocabFromSlice(b *testing.B) {
	tokens := make([]string, 30000)
	for i := range tokens {
		tokens[i] = "token_" + string(rune(i%1000+'a'))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = transformers.VocabFromSlice(tokens)
	}
}

func BenchmarkVocabNew(b *testing.B) {
	tokens := make([]string, 30000)
	for i := range tokens {
		tokens[i] = "token_" + string(rune(i%1000+'a'))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = transformers.New(tokens)
	}
}

func BenchmarkVocabAdd(b *testing.B) {
	vocab := transformers.New([]string{})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vocab.Add("token")
	}
}

func BenchmarkVocabIsInVocab(b *testing.B) {
	vocab, err := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	if err != nil {
		b.Fatalf("Failed to load vocab: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = vocab.IsInVocab("的")
	}
}

// --- Utility Benchmarks ---

func BenchmarkIsDefaultSpecial(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = transformers.IsDefaultSpecial("[UNK]")
		_ = transformers.IsDefaultSpecial("[CLS]")
		_ = transformers.IsDefaultSpecial("[SEP]")
		_ = transformers.IsDefaultSpecial("[MASK]")
		_ = transformers.IsDefaultSpecial("normal")
	}
}

func BenchmarkGroupPiecesFromTokens(b *testing.B) {
	tokens := []transformers.StringOffsetsPair{
		{String: "hello", Offsets: transformers.OffsetsType{Start: 0, End: 5}},
		{String: "##wor", Offsets: transformers.OffsetsType{Start: 5, End: 8}},
		{String: "##ld", Offsets: transformers.OffsetsType{Start: 8, End: 10}},
		{String: "test", Offsets: transformers.OffsetsType{Start: 11, End: 15}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = transformers.GroupPieces(tokens)
	}
}

func BenchmarkMakeOffsetPairsFromGroups(b *testing.B) {
	tokens := []transformers.StringOffsetsPair{
		{String: "hello", Offsets: transformers.OffsetsType{Start: 0, End: 5}},
		{String: "##wor", Offsets: transformers.OffsetsType{Start: 5, End: 8}},
		{String: "##ld", Offsets: transformers.OffsetsType{Start: 8, End: 10}},
		{String: "test", Offsets: transformers.OffsetsType{Start: 11, End: 15}},
	}
	groups := transformers.GroupPieces(tokens)
	text := "helloworldtest"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = transformers.MakeOffsetPairsFromGroups(text, tokens, groups)
	}
}

func BenchmarkGetStrings_Small(b *testing.B) {
	tokens := []transformers.StringOffsetsPair{
		{String: "hello", Offsets: transformers.OffsetsType{Start: 0, End: 5}},
		{String: "world", Offsets: transformers.OffsetsType{Start: 6, End: 11}},
		{String: "test", Offsets: transformers.OffsetsType{Start: 12, End: 16}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = transformers.GetStrings(tokens)
	}
}

func BenchmarkGetOffsets(b *testing.B) {
	tokens := []transformers.StringOffsetsPair{
		{String: "hello", Offsets: transformers.OffsetsType{Start: 0, End: 5}},
		{String: "world", Offsets: transformers.OffsetsType{Start: 6, End: 11}},
		{String: "test", Offsets: transformers.OffsetsType{Start: 12, End: 16}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = transformers.GetOffsets(tokens)
	}
}
