package transformers_test

import (
	"os"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/models/transformers"
)

func TestVocabFromFile_Success(t *testing.T) {
	vocab, err := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	if err != nil {
		t.Fatalf("VocabFromFile failed: %v", err)
	}
	if vocab.Size() == 0 {
		t.Fatal("vocab size should not be zero")
	}
}

func TestVocabFromFile_NotExist(t *testing.T) {
	_, err := transformers.VocabFromFile("../test/nonexistent-vocab.txt")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
	if !os.IsNotExist(err) {
		t.Fatalf("expected file not exist error, got: %v", err)
	}
}

func TestVocabFromSlice_Success(t *testing.T) {
	tokens := []string{"hello", "world", "test"}
	vocab, err := transformers.VocabFromSlice(tokens)
	if err != nil {
		t.Fatalf("VocabFromSlice failed: %v", err)
	}
	if vocab.Size() != 3 {
		t.Fatalf("expected size 3, got %d", vocab.Size())
	}
}

func TestVocabFromSlice_EmptySlice(t *testing.T) {
	tokens := []string{}
	_, err := transformers.VocabFromSlice(tokens)
	if err == nil {
		t.Fatal("expected error for empty slice")
	}
}

func TestNew_Success(t *testing.T) {
	tokens := []string{"apple", "banana", "cherry"}
	vocab := transformers.New(tokens)
	if vocab.Size() != 3 {
		t.Fatalf("expected size 3, got %d", vocab.Size())
	}
	if vocab.GetID("apple") != 0 {
		t.Fatalf("expected apple ID to be 0, got %d", vocab.GetID("apple"))
	}
	if vocab.GetID("banana") != 1 {
		t.Fatalf("expected banana ID to be 1, got %d", vocab.GetID("banana"))
	}
	if vocab.GetID("cherry") != 2 {
		t.Fatalf("expected cherry ID to be 2, got %d", vocab.GetID("cherry"))
	}
}

func TestDict_Add(t *testing.T) {
	vocab := transformers.New([]string{"first"})
	vocab.Add("second")
	vocab.Add("third")
	if vocab.Size() != 3 {
		t.Fatalf("expected size 3, got %d", vocab.Size())
	}
	if vocab.GetID("second") != 1 {
		t.Fatalf("expected second ID to be 1, got %d", vocab.GetID("second"))
	}
	if vocab.GetID("third") != 2 {
		t.Fatalf("expected third ID to be 2, got %d", vocab.GetID("third"))
	}
}

func TestDictGetID_Exists(t *testing.T) {
	vocab := transformers.New([]string{"hello", "world"})
	id := vocab.GetID("hello")
	if id != 0 {
		t.Fatalf("expected ID 0, got %d", id)
	}
}

func TestDictGetID_NotExists(t *testing.T) {
	vocab := transformers.New([]string{"hello", "world"})
	id := vocab.GetID("nonexistent")
	if id != -1 {
		t.Fatalf("expected ID -1 for nonexistent token, got %d", id)
	}
}

func TestDictSize(t *testing.T) {
	tests := []struct {
		name   string
		tokens []string
		want   int
	}{
		{"empty", []string{}, 0},
		{"single", []string{"one"}, 1},
		{"multiple", []string{"a", "b", "c", "d"}, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vocab := transformers.New(tt.tokens)
			if got := vocab.Size(); got != tt.want {
				t.Errorf("Dict.Size() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDictLongestSubstring_Match(t *testing.T) {
	vocab := transformers.New([]string{"hello", "world", "he", "lo"})
	tests := []struct {
		name  string
		token string
		want  string
	}{
		{"full match", "hello", "hello"},
		{"partial match", "hello world", "hello"},
		{"shorter match", "he", "he"},
		{"longest match", "hello world", "hello"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := vocab.LongestSubstring(tt.token); got != tt.want {
				t.Errorf("Dict.LongestSubstring() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDictLongestSubstring_NoMatch(t *testing.T) {
	vocab := transformers.New([]string{"hello", "world"})
	result := vocab.LongestSubstring("xyz")
	if result != "" {
		t.Fatalf("expected empty string, got %q", result)
	}
}

func TestDictConvertItems(t *testing.T) {
	vocab := transformers.New([]string{"a", "b", "c"})
	items := []string{"a", "c", "b", "a"}
	ids := vocab.ConvertItems(items)
	expected := []transformers.ID{0, 2, 1, 0}
	if len(ids) != len(expected) {
		t.Fatalf("expected length %d, got %d", len(expected), len(ids))
	}
	for i := range expected {
		if ids[i] != expected[i] {
			t.Errorf("at index %d: expected %d, got %d", i, expected[i], ids[i])
		}
	}
}

func TestDictConvertTokens(t *testing.T) {
	vocab := transformers.New([]string{"x", "y", "z"})
	tokens := []string{"x", "y", "z", "x"}
	ids1 := vocab.ConvertTokens(tokens)
	ids2 := vocab.ConvertItems(tokens)
	if len(ids1) != len(ids2) {
		t.Fatalf("ConvertTokens and ConvertItems should return same length")
	}
	for i := range ids1 {
		if ids1[i] != ids2[i] {
			t.Errorf("at index %d: ConvertTokens = %d, ConvertItems = %d", i, ids1[i], ids2[i])
		}
	}
}

func TestDictIsInVocab_True(t *testing.T) {
	vocab := transformers.New([]string{"apple", "banana"})
	if !vocab.IsInVocab("apple") {
		t.Fatal("expected apple to be in vocab")
	}
}

func TestDictIsInVocab_False(t *testing.T) {
	vocab := transformers.New([]string{"apple", "banana"})
	if vocab.IsInVocab("orange") {
		t.Fatal("expected orange to not be in vocab")
	}
}

func TestIDInt64(t *testing.T) {
	tests := []struct {
		name string
		id   transformers.ID
		want int64
	}{
		{"zero", 0, 0},
		{"positive", 100, 100},
		{"negative", -1, -1},
		{"large", 2147483647, 2147483647},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.id.Int64(); got != tt.want {
				t.Errorf("ID.Int64() = %v, want %v", got, tt.want)
			}
		})
	}
}

func BenchmarkVocabFromFile(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	}
}

func BenchmarkVocabGetID(b *testing.B) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = vocab.GetID("的")
	}
}

func BenchmarkVocabLongestSubstring(b *testing.B) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = vocab.LongestSubstring("这是一个测试字符串")
	}
}

func BenchmarkVocabConvertItems(b *testing.B) {
	vocab, _ := transformers.VocabFromFile("../../test/bert-chinese-vocab.txt")
	tokens := make([]string, 100)
	for i := range tokens {
		tokens[i] = "的"
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = vocab.ConvertItems(tokens)
	}
}
