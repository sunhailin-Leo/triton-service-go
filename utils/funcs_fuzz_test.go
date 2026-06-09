package utils_test

import (
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

// FuzzIsWhitespace fuzzes the IsWhitespace function.
func FuzzIsWhitespace(f *testing.F) {
	seedRunes := []rune{' ', '\t', '\n', '\r', '\u2003', '\u3000', 'a', '1', ',', '中', 0x00, 0x1F, 0x7F}
	for _, r := range seedRunes {
		f.Add(uint32(r))
	}
	f.Fuzz(func(t *testing.T, r uint32) {
		_ = utils.IsWhitespace(rune(r))
	})
}

// FuzzIsControl fuzzes the IsControl function.
func FuzzIsControl(f *testing.F) {
	seedRunes := []rune{'\t', '\n', '\r', 0x00, 0x07, 0x1B, 'a', '1', ' ', 0x7F, 0x80, 0x9F}
	for _, r := range seedRunes {
		f.Add(uint32(r))
	}
	f.Fuzz(func(t *testing.T, r uint32) {
		// Tab, newline, carriage return are NOT control chars per BERT definition
		result := utils.IsControl(rune(r))
		if r == '\t' || r == '\n' || r == '\r' {
			if result {
				t.Errorf("IsControl(%q) = true, want false", rune(r))
			}
		}
	})
}

// FuzzIsPunctuation fuzzes the IsPunctuation function.
func FuzzIsPunctuation(f *testing.F) {
	seedRunes := []rune{',', '.', '!', '?', ':', ';', '-', '(', ')', 'a', '1', ' ', '中', 0x00}
	for _, r := range seedRunes {
		f.Add(uint32(r))
	}
	f.Fuzz(func(t *testing.T, r uint32) {
		_ = utils.IsPunctuation(rune(r))
	})
}

// FuzzIsChinese fuzzes the IsChinese function.
func FuzzIsChinese(f *testing.F) {
	seedRunes := []rune{'中', '文', 'a', '1', ',', ' ', 'あ', '가', 0x4E00, 0x9FFF, 0x3400, 0x4DBF}
	for _, r := range seedRunes {
		f.Add(uint32(r))
	}
	f.Fuzz(func(t *testing.T, r uint32) {
		_ = utils.IsChinese(rune(r))
	})
}

// FuzzIsChineseOrNumber fuzzes the IsChineseOrNumber function.
func FuzzIsChineseOrNumber(f *testing.F) {
	seedRunes := []rune{'中', '0', '9', 'a', 'Z', ',', ' ', 'あ'}
	for _, r := range seedRunes {
		f.Add(uint32(r))
	}
	f.Fuzz(func(t *testing.T, r uint32) {
		result := utils.IsChineseOrNumber(rune(r))
		// Digits 0-9 must always return true
		if r >= '0' && r <= '9' {
			if !result {
				t.Errorf("IsChineseOrNumber(%q) = false for digit, want true", rune(r))
			}
		}
	})
}

// FuzzClean fuzzes the Clean function.
func FuzzClean(f *testing.F) {
	seeds := []string{
		"hello world",
		"hello\x00world",
		"hello\t\nworld",
		"",
		"\x00\x01\x07",
		"正常文本",
		"hello\ufffdworld",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		result := utils.Clean(text)
		// Clean should never return a string containing \x00, \ufffd, or control chars (except space)
		for _, r := range result {
			if r == '\x00' || r == '\ufffd' {
				t.Errorf("Clean() result contains invalid rune: %q", r)
			}
			if r != ' ' && utils.IsControl(r) {
				t.Errorf("Clean() result contains control rune: %q", r)
			}
		}
	})
}

// FuzzPadChinese fuzzes the PadChinese function.
func FuzzPadChinese(f *testing.F) {
	seeds := []string{
		"中文",
		"hello",
		"中文English",
		"中123",
		"",
		"hello world",
		"中，文",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		result := utils.PadChinese(text)
		// PadChinese should never panic
		_ = result
	})
}

// FuzzStripAccentsAndLower fuzzes the StripAccentsAndLower function.
func FuzzStripAccentsAndLower(f *testing.F) {
	seeds := []string{
		"Héllo Wórld",
		"áéíóú",
		"",
		"NORMAL",
		"123",
		"café",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		result := utils.StripAccentsAndLower(text)
		// Result should always be lowercase
		for _, r := range result {
			if r >= 'A' && r <= 'Z' {
				t.Errorf("StripAccentsAndLower() result contains uppercase: %q", r)
			}
		}
	})
}

// FuzzSplitPunctuation fuzzes the SplitPunctuation function.
func FuzzSplitPunctuation(f *testing.F) {
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
		// SplitPunctuation should never panic
		_ = result
	})
}

// FuzzBinaryFilter fuzzes the BinaryFilter function.
func FuzzBinaryFilter(f *testing.F) {
	seeds := [][]byte{
		{1, 2, 3},
		{0, 0, 0},
		{},
		{1, 0, 2, 0, 3},
		{255, 0, 128},
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, arr []byte) {
		result := utils.BinaryFilter(arr)
		// Result should never contain 0x00 bytes
		for _, b := range result {
			if b == 0 {
				t.Error("BinaryFilter() result contains zero byte")
			}
		}
	})
}

// FuzzBinaryToSlice fuzzes the BinaryToSlice function.
func FuzzBinaryToSlice(f *testing.F) {
	f.Add([]byte{}, 4, utils.TritonINT32Type)
	f.Add(makeInt32Bytes([]int32{1, 2, 3}), 4, utils.TritonINT32Type)
	f.Add(makeInt64Bytes([]int64{1, 2}), 8, utils.TritonINT64Type)
	f.Add(makeFloat32Bytes([]float32{1.0, 2.0}), 4, utils.SliceFloat32Type)
	f.Add(makeFloat64Bytes([]float64{1.0}), 8, utils.SliceFloat64Type)
	f.Add(makeFloat64Bytes([]float64{1.0}), 8, utils.TritonFP64Type)
	f.Add(makeFloat32Bytes([]float32{1.0}), 4, utils.TritonFP32Type)
	f.Add(makeInt64Bytes([]int64{1}), 8, utils.SliceInt64Type)
	f.Add(makeInt32Bytes([]int32{1}), 4, utils.SliceIntType)
	f.Add([]byte("hello"), 4, utils.TritonBytesType)
	f.Add([]byte("test"), 4, utils.SliceByteType)
	f.Add(makeFloat32Bytes([]float32{1.0}), 4, utils.TritonFP16Type)

	f.Fuzz(func(t *testing.T, body []byte, bytesLen int, returnType string) {
		// BinaryToSlice should never panic, even with invalid input
		if bytesLen <= 0 {
			return
		}
		if len(body) == 0 {
			return
		}
		result := utils.BinaryToSlice(body, bytesLen, returnType)
		_ = result
	})
}

// FuzzBinaryToSlice_INT32_Roundtrip verifies INT32 roundtrip encoding/decoding.
func FuzzBinaryToSlice_INT32_Roundtrip(f *testing.F) {
	f.Add("0,1,-1,100,-100")
	f.Add("42")
	f.Add("")
	f.Fuzz(func(t *testing.T, csv string) {
		values := parseInt32CSV(csv)
		body := makeInt32Bytes(values)
		result := utils.BinaryToSlice(body, 4, utils.TritonINT32Type)
		if len(result) != len(values) {
			t.Errorf("roundtrip length mismatch: got %d, want %d", len(result), len(values))
			return
		}
		for i, v := range values {
			if result[i].(int32) != v {
				t.Errorf("roundtrip value mismatch at %d: got %d, want %d", i, result[i].(int32), v)
			}
		}
	})
}

// FuzzBinaryToSlice_FP32_Roundtrip verifies FP32 roundtrip encoding/decoding.
func FuzzBinaryToSlice_FP32_Roundtrip(f *testing.F) {
	f.Add("0,1,-1,3.14")
	f.Add("42")
	f.Add("")
	f.Fuzz(func(t *testing.T, csv string) {
		values := parseFloat32CSV(csv)
		body := makeFloat32Bytes(values)
		result := utils.BinaryToSlice(body, 4, utils.SliceFloat32Type)
		if len(result) != len(values) {
			t.Errorf("roundtrip length mismatch: got %d, want %d", len(result), len(values))
			return
		}
		for i, v := range values {
			got := result[i].(float32)
			if math.Float32bits(got) != math.Float32bits(v) {
				t.Errorf("roundtrip value mismatch at %d: got %v, want %v", i, got, v)
			}
		}
	})
}

// FuzzBinaryToSlice_FP64_Roundtrip verifies FP64 roundtrip encoding/decoding.
func FuzzBinaryToSlice_FP64_Roundtrip(f *testing.F) {
	f.Add("0,1,-1,3.14")
	f.Add("42")
	f.Add("")
	f.Fuzz(func(t *testing.T, csv string) {
		values := parseFloat64CSV(csv)
		body := makeFloat64Bytes(values)
		result := utils.BinaryToSlice(body, 8, utils.SliceFloat64Type)
		if len(result) != len(values) {
			t.Errorf("roundtrip length mismatch: got %d, want %d", len(result), len(values))
			return
		}
		for i, v := range values {
			got := result[i].(float64)
			if math.Float64bits(got) != math.Float64bits(v) {
				t.Errorf("roundtrip value mismatch at %d: got %v, want %v", i, got, v)
			}
		}
	})
}

// FuzzBinaryToSlice_INT64_Roundtrip verifies INT64 roundtrip encoding/decoding.
func FuzzBinaryToSlice_INT64_Roundtrip(f *testing.F) {
	f.Add("0,1,-1,100")
	f.Add("42")
	f.Add("")
	f.Fuzz(func(t *testing.T, csv string) {
		values := parseInt64CSV(csv)
		body := makeInt64Bytes(values)
		result := utils.BinaryToSlice(body, 8, utils.TritonINT64Type)
		if len(result) != len(values) {
			t.Errorf("roundtrip length mismatch: got %d, want %d", len(result), len(values))
			return
		}
		for i, v := range values {
			if result[i].(int64) != v {
				t.Errorf("roundtrip value mismatch at %d: got %d, want %d", i, result[i].(int64), v)
			}
		}
	})
}

// parseInt32CSV parses a comma-separated string into []int32.
func parseInt32CSV(csv string) []int32 {
	if csv == "" {
		return nil
	}
	parts := strings.Split(csv, ",")
	result := make([]int32, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseInt(p, 10, 32)
		if err != nil {
			continue
		}
		result = append(result, int32(v))
	}
	return result
}

// parseInt64CSV parses a comma-separated string into []int64.
func parseInt64CSV(csv string) []int64 {
	if csv == "" {
		return nil
	}
	parts := strings.Split(csv, ",")
	result := make([]int64, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseInt(p, 10, 64)
		if err != nil {
			continue
		}
		result = append(result, v)
	}
	return result
}

// parseFloat32CSV parses a comma-separated string into []float32.
func parseFloat32CSV(csv string) []float32 {
	if csv == "" {
		return nil
	}
	parts := strings.Split(csv, ",")
	result := make([]float32, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseFloat(p, 32)
		if err != nil {
			continue
		}
		result = append(result, float32(v))
	}
	return result
}

// parseFloat64CSV parses a comma-separated string into []float64.
func parseFloat64CSV(csv string) []float64 {
	if csv == "" {
		return nil
	}
	parts := strings.Split(csv, ",")
	result := make([]float64, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseFloat(p, 64)
		if err != nil {
			continue
		}
		result = append(result, v)
	}
	return result
}
