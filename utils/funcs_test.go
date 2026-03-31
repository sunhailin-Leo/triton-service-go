package utils_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

func TestIsWhitespace(t *testing.T) {
	tests := []struct {
		name     string
		c        rune
		expected bool
	}{
		{"space", ' ', true},
		{"tab", '\t', true},
		{"newline", '\n', true},
		{"carriage return", '\r', true},
		{"normal character", 'a', false},
		{"digit", '1', false},
		{"unicode Zs - em space", '\u2003', true},
		{"unicode Zs - ideographic space", '\u3000', true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.IsWhitespace(tt.c)
			if result != tt.expected {
				t.Errorf("IsWhitespace(%q) = %v, want %v", tt.c, result, tt.expected)
			}
		})
	}
}

func TestIsControl(t *testing.T) {
	tests := []struct {
		name     string
		c        rune
		expected bool
	}{
		{"tab", '\t', false},
		{"newline", '\n', false},
		{"carriage return", '\r', false},
		{"control char null", '\x00', true},
		{"control char bell", '\x07', true},
		{"control char escape", '\x1b', true},
		{"normal character", 'a', false},
		{"digit", '1', false},
		{"space", ' ', false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.IsControl(tt.c)
			if result != tt.expected {
				t.Errorf("IsControl(%q) = %v, want %v", tt.c, result, tt.expected)
			}
		})
	}
}

func TestIsPunctuation(t *testing.T) {
	tests := []struct {
		name     string
		c        rune
		expected bool
	}{
		{"comma", ',', true},
		{"period", '.', true},
		{"question mark", '?', true},
		{"exclamation", '!', true},
		{"colon", ':', true},
		{"semicolon", ';', true},
		{"hyphen", '-', true},
		{"space", ' ', false},
		{"letter", 'a', false},
		{"digit", '1', false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.IsPunctuation(tt.c)
			if result != tt.expected {
				t.Errorf("IsPunctuation(%q) = %v, want %v", tt.c, result, tt.expected)
			}
		})
	}
}

func TestIsChinese(t *testing.T) {
	tests := []struct {
		name     string
		c        rune
		expected bool
	}{
		{"simple chinese", '中', true},
		{"chinese char range 1", '㐀', false},
		{"chinese char range 2", '䶿', false},
		{"japanese", 'あ', false},
		{"korean", '가', false},
		{"english", 'a', false},
		{"digit", '1', false},
		{"punctuation", '，', true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.IsChinese(tt.c)
			if result != tt.expected {
				t.Errorf("IsChinese(%q) = %v, want %v", tt.c, result, tt.expected)
			}
		})
	}
}

func TestIsChineseOrNumber(t *testing.T) {
	tests := []struct {
		name     string
		c        rune
		expected bool
	}{
		{"chinese char", '中', true},
		{"digit", '1', true},
		{"digit zero", '0', true},
		{"digit nine", '9', true},
		{"english letter", 'a', false},
		{"english letter upper", 'A', false},
		{"space", ' ', false},
		{"punctuation", ',', true},
		{"chinese punctuation", '，', true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.IsChineseOrNumber(tt.c)
			if result != tt.expected {
				t.Errorf("IsChineseOrNumber(%q) = %v, want %v", tt.c, result, tt.expected)
			}
		})
	}
}

func TestIsWhiteSpaceOrChinese(t *testing.T) {
	tests := []struct {
		name     string
		c        rune
		expected bool
	}{
		{"space", ' ', true},
		{"tab", '\t', true},
		{"chinese char", '中', true},
		{"english letter", 'a', false},
		{"digit", '1', false},
		{"punctuation", ',', true},
		{"chinese punctuation", '，', true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.IsWhiteSpaceOrChinese(tt.c)
			if result != tt.expected {
				t.Errorf("IsWhiteSpaceOrChinese(%q) = %v, want %v", tt.c, result, tt.expected)
			}
		})
	}
}

func TestIsWhiteSpaceOrChineseOrNumber(t *testing.T) {
	tests := []struct {
		name     string
		c        rune
		expected bool
	}{
		{"space", ' ', true},
		{"tab", '\t', true},
		{"chinese char", '中', true},
		{"digit", '1', true},
		{"english letter", 'a', false},
		{"punctuation", ',', true},
		{"chinese punctuation", '，', true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.IsWhiteSpaceOrChineseOrNumber(tt.c)
			if result != tt.expected {
				t.Errorf("IsWhiteSpaceOrChineseOrNumber(%q) = %v, want %v", tt.c, result, tt.expected)
			}
		})
	}
}

func TestClean(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		expected string
	}{
		{"remove null char", "hello\x00world", "helloworld"},
		{"remove replacement char", "hello\ufffdworld", "helloworld"},
		{"remove control char", "hello\x07world", "helloworld"},
		{"replace whitespace", "hello\tworld", "hello world"},
		{"replace newline", "hello\nworld", "hello world"},
		{"replace multiple whitespace", "hello\t\nworld", "hello  world"},
		{"keep normal text", "hello world", "hello world"},
		{"mixed control and normal", "hello\x00\tworld\n\x07test", "hello world test"},
		{"empty string", "", ""},
		{"only control chars", "\x00\x07", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.Clean(tt.text)
			if result != tt.expected {
				t.Errorf("Clean(%q) = %q, want %q", tt.text, result, tt.expected)
			}
		})
	}
}

func TestPadChinese(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		expected string
	}{
		{"single chinese", "中", " 中 "},
		{"chinese with english", "中English", " 中 English"},
		{"chinese with digit", "中123", " 中 123"},
		{"multiple chinese", "中文测试", " 中  文  测  试 "},
		{"mixed text", "Hello中文World", "Hello 中  文 World"},
		{"no chinese", "Hello World", "Hello World"},
		{"empty string", "", ""},
		{"chinese with punctuation", "中，文", " 中  ，  文 "},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.PadChinese(tt.text)
			if result != tt.expected {
				t.Errorf("PadChinese(%q) = %q, want %q", tt.text, result, tt.expected)
			}
		})
	}
}

func TestCleanAndPadChineseWithWhiteSpace(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		expected []string
	}{
		{"simple chinese", "中文测试", []string{"中", "文", "测", "试"}},
		{"mixed with control", "hello\x00中", []string{"hello", "中"}},
		{"with whitespace", "hello 中 world", []string{"hello", "中", "world"}},
		{"with newline", "hello\n中", []string{"hello", "中"}},
		{"empty string", "", []string{}},
		{"only control", "\x00\x07", []string{}},
		{"multiple spaces", "hello  中", []string{"hello", "中"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.CleanAndPadChineseWithWhiteSpace(tt.text)
			if len(result) != len(tt.expected) {
				t.Errorf("CleanAndPadChineseWithWhiteSpace(%q) = %v, want %v", tt.text, result, tt.expected)
			} else {
				for i := range result {
					if result[i] != tt.expected[i] {
						t.Errorf("CleanAndPadChineseWithWhiteSpace(%q)[%d] = %q, want %q", tt.text, i, result[i], tt.expected[i])
					}
				}
			}
		})
	}
}

func TestStripAccentsAndLower(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		expected string
	}{
		{"accent a", "á", "a"},
		{"accent e", "é", "e"},
		{"accent i", "í", "i"},
		{"accent o", "ó", "o"},
		{"accent u", "ú", "u"},
		{"accent c", "ç", "c"},
		{"accent n", "ñ", "n"},
		{"mixed accents", "áéíóú", "aeiou"},
		{"uppercase with accent", "Á", "a"},
		{"normal text", "Hello", "hello"},
		{"mixed", "Héllo Wórld", "hello world"},
		{"empty string", "", ""},
		{"no accents", "hello world", "hello world"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.StripAccentsAndLower(tt.text)
			if result != tt.expected {
				t.Errorf("StripAccentsAndLower(%q) = %q, want %q", tt.text, result, tt.expected)
			}
		})
	}
}

func TestSplitPunctuation(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		expected []string
	}{
		{"simple punctuation", "hello,world", []string{"hello", ",", "world"}},
		{"multiple punctuation", "hello,world!", []string{"hello", ",", "world", "!"}},
		{"no punctuation", "hello world", []string{"hello world"}},
		{"only punctuation", ",!", []string{"", ",", "", "!"}},
		{"empty string", "", []string(nil)},
		{"punctuation at start", "!hello", []string{"", "!", "hello"}},
		{"punctuation at end", "hello!", []string{"hello", "!"}},
		{"mixed", "hello, world!", []string{"hello", ",", " world", "!"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.SplitPunctuation(tt.text)
			if len(result) != len(tt.expected) {
				t.Errorf("SplitPunctuation(%q) = %v, want %v", tt.text, result, tt.expected)
			} else {
				for i := range result {
					if result[i] != tt.expected[i] {
						t.Errorf("SplitPunctuation(%q)[%d] = %q, want %q", tt.text, i, result[i], tt.expected[i])
					}
				}
			}
		})
	}
}

func TestBinaryFilter(t *testing.T) {
	tests := []struct {
		name     string
		arr      []byte
		expected []byte
	}{
		{"filter zero bytes", []byte{1, 0, 2, 0, 3}, []byte{32, 3}},
		{"replace zero with space", []byte{1, 0, 0, 2}, []byte{2}},
		{"no zero bytes", []byte{1, 2, 3}, []byte{1, 2, 3}},
		{"all zeros", []byte{0, 0, 0}, []byte{}},
		{"zero at start", []byte{0, 1, 2}, []byte{1, 2}},
		{"zero at end", []byte{1, 2, 0}, []byte{1, 32}},
		{"empty array", []byte{}, []byte{}},
		{"single zero", []byte{0}, []byte{}},
		{"multiple consecutive zeros", []byte{1, 0, 0, 0, 2}, []byte{2}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.BinaryFilter(tt.arr)
			if len(result) != len(tt.expected) {
				t.Errorf("BinaryFilter(%v) = %v, want %v", tt.arr, result, tt.expected)
			} else {
				for i := range result {
					if result[i] != tt.expected[i] {
						t.Errorf("BinaryFilter(%v)[%d] = %v, want %v", tt.arr, i, result[i], tt.expected[i])
					}
				}
			}
		})
	}
}

func TestBinaryToSlice(t *testing.T) {
	tests := []struct {
		name       string
		body       []byte
		bytesLen   int
		returnType string
		validate   func(t *testing.T, result []interface{})
	}{
		{
			name:       "INT32 type",
			body:       makeInt32Bytes([]int32{1, 2, 3}),
			bytesLen:   4,
			returnType: utils.TritonINT32Type,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 3 {
					t.Errorf("expected 3 elements, got %d", len(result))
				}
				if result[0].(int32) != 1 || result[1].(int32) != 2 || result[2].(int32) != 3 {
					t.Errorf("unexpected values: %v", result)
				}
			},
		},
		{
			name:       "INT64 type",
			body:       makeInt64Bytes([]int64{1, 2, 3}),
			bytesLen:   8,
			returnType: utils.TritonINT64Type,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 3 {
					t.Errorf("expected 3 elements, got %d", len(result))
				}
				if result[0].(int64) != 1 || result[1].(int64) != 2 || result[2].(int64) != 3 {
					t.Errorf("unexpected values: %v", result)
				}
			},
		},
		{
			name:       "SliceFloat32 type",
			body:       makeFloat32Bytes([]float32{1.0, 2.0, 3.0}),
			bytesLen:   4,
			returnType: utils.SliceFloat32Type,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 3 {
					t.Errorf("expected 3 elements, got %d", len(result))
				}
				if result[0].(float32) != 1.0 || result[1].(float32) != 2.0 || result[2].(float32) != 3.0 {
					t.Errorf("unexpected values: %v", result)
				}
			},
		},
		{
			name:       "TritonFP16 type",
			body:       makeFloat32Bytes([]float32{1.0, 2.0, 3.0}),
			bytesLen:   4,
			returnType: utils.TritonFP16Type,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 3 {
					t.Errorf("expected 3 elements, got %d", len(result))
				}
			},
		},
		{
			name:       "SliceFloat64 type",
			body:       makeFloat32Bytes([]float32{1.0, 2.0, 3.0}),
			bytesLen:   4,
			returnType: utils.SliceFloat64Type,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 3 {
					t.Errorf("expected 3 elements, got %d", len(result))
				}
			},
		},
		{
			name:       "TritonFP32 type",
			body:       makeFloat32Bytes([]float32{1.0, 2.0, 3.0}),
			bytesLen:   4,
			returnType: utils.TritonFP32Type,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 3 {
					t.Errorf("expected 3 elements, got %d", len(result))
				}
			},
		},
		{
			name:       "SliceInt64 type",
			body:       makeInt32Bytes([]int32{1, 2, 3}),
			bytesLen:   4,
			returnType: utils.SliceInt64Type,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 3 {
					t.Errorf("expected 3 elements, got %d", len(result))
				}
			},
		},
		{
			name:       "SliceInt type",
			body:       makeInt32Bytes([]int32{1, 2, 3}),
			bytesLen:   4,
			returnType: utils.SliceIntType,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 3 {
					t.Errorf("expected 3 elements, got %d", len(result))
				}
			},
		},
		{
			name:       "TritonBytes type",
			body:       []byte("hello world"),
			bytesLen:   4,
			returnType: utils.TritonBytesType,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 2 {
					t.Errorf("expected 2 elements, got %d", len(result))
				}
			},
		},
		{
			name:       "SliceByte type",
			body:       []byte("hello world"),
			bytesLen:   4,
			returnType: utils.SliceByteType,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 2 {
					t.Errorf("expected 2 elements, got %d", len(result))
				}
			},
		},
		{
			name:       "empty input",
			body:       []byte{},
			bytesLen:   4,
			returnType: utils.TritonINT32Type,
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 0 {
					t.Errorf("expected 0 elements, got %d", len(result))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.BinaryToSlice(tt.body, tt.bytesLen, tt.returnType)
			tt.validate(t, result)
		})
	}

	// Incomplete bytes causes panic in BinaryToSlice due to binary.LittleEndian.Uint32
	t.Run("incomplete bytes panics", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for incomplete bytes, but did not panic")
			}
		}()
		incompleteBody := append(makeInt32Bytes([]int32{1, 2}), 0x01)
		utils.BinaryToSlice(incompleteBody, 4, utils.TritonINT32Type)
	})
}

func makeInt32Bytes(values []int32) []byte {
	result := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(result[i*4:], uint32(v))
	}
	return result
}

func makeInt64Bytes(values []int64) []byte {
	result := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(result[i*8:], uint64(v))
	}
	return result
}

func makeFloat32Bytes(values []float32) []byte {
	result := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(result[i*4:], math.Float32bits(v))
	}
	return result
}

func BenchmarkClean(b *testing.B) {
	text := "hello\x00world\ttest\n\x07data"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.Clean(text)
	}
}

func BenchmarkStripAccentsAndLower(b *testing.B) {
	text := "Héllo Wórld with áccénts"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.StripAccentsAndLower(text)
	}
}

func BenchmarkBinaryToSlice_INT32(b *testing.B) {
	body := makeInt32Bytes(makeInt32Slice(128))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.BinaryToSlice(body, 4, utils.TritonINT32Type)
	}
}

func BenchmarkBinaryToSlice_FP32(b *testing.B) {
	body := makeFloat32Bytes(makeFloat32Slice(128))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.BinaryToSlice(body, 4, utils.SliceFloat32Type)
	}
}

func BenchmarkIsChinese(b *testing.B) {
	c := '中'
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.IsChinese(c)
	}
}

func BenchmarkIsPunctuation(b *testing.B) {
	c := ','
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.IsPunctuation(c)
	}
}

func makeInt32Slice(n int) []int32 {
	result := make([]int32, n)
	for i := 0; i < n; i++ {
		result[i] = int32(i)
	}
	return result
}

func makeFloat32Slice(n int) []float32 {
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		result[i] = float32(i)
	}
	return result
}

// BenchmarkPadChinese benchmarks the PadChinese function
func BenchmarkPadChinese(b *testing.B) {
	text := "这是一个测试句子，用于基准测试中文分词的性能。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		utils.PadChinese(text)
	}
}

// BenchmarkPadChinese_Long benchmarks PadChinese with longer text
func BenchmarkPadChinese_Long(b *testing.B) {
	text := "这是一个较长的测试句子，包含更多的中文字符，用于测试在更长文本情况下PadChinese函数的性能表现。"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		utils.PadChinese(text)
	}
}

// BenchmarkBinaryFilter benchmarks the BinaryFilter function
func BenchmarkBinaryFilter(b *testing.B) {
	arr := make([]byte, 256)
	for i := range arr {
		if i%5 == 0 {
			arr[i] = 0
		} else {
			arr[i] = byte(i)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		utils.BinaryFilter(arr)
	}
}

// BenchmarkBinaryFilter_Sparse benchmarks BinaryFilter with sparse zeros
func BenchmarkBinaryFilter_Sparse(b *testing.B) {
	arr := make([]byte, 512)
	for i := range arr {
		if i%10 == 0 {
			arr[i] = 0
		} else {
			arr[i] = byte(i)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		utils.BinaryFilter(arr)
	}
}

// BenchmarkBinaryToSlice_INT64 benchmarks BinaryToSlice for INT64 type
func BenchmarkBinaryToSlice_INT64(b *testing.B) {
	body := makeInt64Bytes(makeInt64Slice(128))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		utils.BinaryToSlice(body, 8, utils.TritonINT64Type)
	}
}

// BenchmarkBinaryToSlice_Bytes benchmarks BinaryToSlice for BYTES type
func BenchmarkBinaryToSlice_Bytes(b *testing.B) {
	body := []byte("hello world test data for benchmarking")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		utils.BinaryToSlice(body, 4, utils.TritonBytesType)
	}
}

// BenchmarkCleanAndPadChineseWithWhiteSpace benchmarks CleanAndPadChineseWithWhiteSpace
func BenchmarkCleanAndPadChineseWithWhiteSpace(b *testing.B) {
	text := "hello\x00world\ttest\n\x07data"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		utils.CleanAndPadChineseWithWhiteSpace(text)
	}
}

// BenchmarkSplitPunctuation benchmarks SplitPunctuation function
func BenchmarkSplitPunctuation(b *testing.B) {
	text := "hello,world!this.is;a:test"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		utils.SplitPunctuation(text)
	}
}

func makeInt64Slice(n int) []int64 {
	result := make([]int64, n)
	for i := 0; i < n; i++ {
		result[i] = int64(i)
	}
	return result
}
