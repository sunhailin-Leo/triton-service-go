package utils

import (
	"encoding/binary"
	"math"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// IsWhitespace checks whether rune c is a BERT whitespace character.
//
//go:inline
func IsWhitespace(c rune) bool {
	if c <= 0xFF {
		return ASCIIWhiteSpace[c]
	}

	return unicode.Is(unicode.Zs, c)
}

// IsControl checks whether rune c is a BERT control character.
//
//go:inline
func IsControl(c rune) bool {
	switch c {
	case '\t':
		return false
	case '\n':
		return false
	case '\r':
		return false
	}

	return unicode.In(c, unicode.Cc, unicode.Cf)
}

// IsPunctuation checks whether rune c is a BERT punctuation character.
//
//go:inline
func IsPunctuation(c rune) bool {
	// return unicode.In(c, utils.Bp, unicode.P)
	// return unicode.In(c, utils.AsciiPunctuation, unicode.P) && c != '-'
	return unicode.In(c, ASCIIPunctuation, unicode.P)
}

// IsChinese validates that rune c is in the CJK range according to BERT spec.
//
//go:inline
func IsChinese(c rune) bool {
	return unicode.In(c, BertChineseChar)
}

// IsChineseOrNumber validates that rune c is in the CJK range according to BERT spec or Number.
//
//go:inline
func IsChineseOrNumber(c rune) bool {
	return unicode.In(c, BertChineseChar) || unicode.IsNumber(c)
}

// IsWhiteSpaceOrChinese validates that rune c is whitespace or is Chinese.
//
//go:inline
func IsWhiteSpaceOrChinese(c rune) bool {
	return IsWhitespace(c) || IsChinese(c)
}

// IsWhiteSpaceOrChineseOrNumber validates that rune c is whitespace or is Chinese or is Number.
//
//go:inline
func IsWhiteSpaceOrChineseOrNumber(c rune) bool {
	return IsWhitespace(c) || IsChineseOrNumber(c)
}

// Clean function will clear some characters.
func Clean(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	for _, c := range text {
		if c == 0 || c == 0xfffd || IsControl(c) {
			continue
		}
		if IsWhitespace(c) {
			b.WriteRune(' ')
		} else {
			b.WriteRune(c)
		}
	}
	return b.String()
}

// PadChinese will add space padding around all CJK chars
// This implementation matches BasicTokenizer._tokenize_chinese_chars.
func PadChinese(text string) string {
	var b strings.Builder
	b.Grow(len(text) + len(text)/2)
	for _, c := range text {
		if IsChinese(c) {
			b.WriteRune(' ')
			b.WriteRune(c)
			b.WriteRune(' ')
		} else {
			b.WriteRune(c)
		}
	}
	return b.String()
}

// CleanAndPadChineseWithWhiteSpace combine three function clean, padChinese, tokenizeWhitespaceV1.
func CleanAndPadChineseWithWhiteSpace(text string) []string {
	var b strings.Builder
	b.Grow(len(text) + len(text)/2)
	for _, c := range text {
		if c == 0 || c == 0xfffd || IsControl(c) {
			continue
		}

		switch {
		case IsChinese(c):
			b.WriteRune(' ')
			b.WriteRune(c)
			b.WriteRune(' ')
		case IsWhitespace(c):
			b.WriteRune(' ')
		default:
			b.WriteRune(c)
		}
	}

	return strings.Fields(strings.TrimSpace(b.String()))
}

// StripAccentsAndLower strip accents and lower.
func StripAccentsAndLower(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	for _, c := range norm.NFD.String(text) {
		if unicode.Is(unicode.Mn, c) {
			continue
		}
		b.WriteRune(unicode.ToLower(c))
		// if !unicode.Is(unicode.Mn, c) {
		//	b.WriteRune(unicode.ToLower(c))
		// }
	}

	return b.String()
}

// SplitPunctuation splits the text around punctuation characters.
func SplitPunctuation(text string) (toks []string) {
	var b strings.Builder
	for _, c := range text {
		if IsPunctuation(c) {
			if b.Len() > 0 {
				toks = append(toks, b.String())
				b.Reset()
			}
			toks = append(toks, string(c))
		} else {
			b.WriteRune(c)
		}
	}
	if b.Len() > 0 {
		toks = append(toks, b.String())
	}
	return
}

// BinaryFilter filters zero bytes from a byte slice, inserting spaces at word boundaries.
func BinaryFilter(arr []byte) []byte {
	result := make([]byte, 0, len(arr))
	for i := range arr {
		if arr[i] == 0 {
			continue
		}
		if i != len(arr)-1 && arr[i+1] == 0 {
			if i != 0 {
				result = append(result, ' ')
			}
			continue
		}
		result = append(result, arr[i])
	}
	return result
}

// fp16ToFloat32 converts a IEEE 754 half-precision (16-bit) value to float32.
func fp16ToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) & 0x1
	exponent := uint32(bits>>10) & 0x1F
	mantissa := uint32(bits) & 0x3FF

	switch exponent {
	case 0:
		if mantissa == 0 {
			return math.Float32frombits(sign << 31)
		}
		// subnormal fp16 → normalized fp32
		for mantissa&0x400 == 0 {
			mantissa <<= 1
			exponent--
		}
		exponent++
		mantissa &= 0x3FF
		return math.Float32frombits((sign << 31) | ((exponent + 112) << 23) | (mantissa << 13))
	case 31:
		// Inf / NaN
		return math.Float32frombits((sign << 31) | 0x7F800000 | (mantissa << 13))
	default:
		return math.Float32frombits((sign << 31) | ((exponent + 112) << 23) | (mantissa << 13))
	}
}

// convert functions.
var convertFuncMap = map[string]func([]uint8) any{
	TritonINT32Type: func(b []uint8) any {
		return int32(binary.LittleEndian.Uint32(b))
	},
	TritonINT64Type: func(b []uint8) any {
		return int64(binary.LittleEndian.Uint64(b))
	},
	TritonFP16Type: func(b []uint8) any {
		return fp16ToFloat32(binary.LittleEndian.Uint16(b))
	},
	TritonFP32Type: func(b []uint8) any {
		return math.Float32frombits(binary.LittleEndian.Uint32(b))
	},
	TritonFP64Type: func(b []uint8) any {
		return math.Float64frombits(binary.LittleEndian.Uint64(b))
	},
	SliceFloat32Type: func(b []uint8) any {
		return math.Float32frombits(binary.LittleEndian.Uint32(b))
	},
	SliceFloat64Type: func(b []uint8) any {
		return math.Float64frombits(binary.LittleEndian.Uint64(b))
	},
	SliceInt64Type: func(b []uint8) any {
		return int64(binary.LittleEndian.Uint64(b))
	},
	SliceIntType: func(b []uint8) any {
		return int(binary.LittleEndian.Uint32(b))
	},
}

// BinaryToSlice converts a byte slice to a typed slice based on the returnType.
// Returns nil if the returnType is not supported.
func BinaryToSlice(body []uint8, bytesLen int, returnType string) []any {
	// special process BYTES and []byte
	if returnType == TritonBytesType || returnType == SliceByteType {
		return SliceToInterfaceSlice(strings.Fields(string(BinaryFilter(body))))
	}
	// response body split by chunk (other types need convert functions)
	convertFunc, ok := convertFuncMap[returnType]
	if !ok || convertFunc == nil {
		return nil
	}
	if bytesLen <= 0 || len(body) == 0 {
		return nil
	}
	convertFuncResult := make([]any, len(body)/bytesLen)
	for i := 0; i < len(convertFuncResult); i++ {
		start := i * bytesLen
		end := start + bytesLen
		if end > len(body) {
			break
		}
		convertFuncResult[i] = convertFunc(body[start:end])
	}
	return convertFuncResult
}
