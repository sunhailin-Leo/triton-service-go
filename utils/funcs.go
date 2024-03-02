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
	// unicode.Is(unicode.Han, c)
	return unicode.In(c, BertChineseChar, unicode.P)
}

// IsChineseOrNumber validates that rune c is in the CJK range according to BERT spec or Number.
//
//go:inline
func IsChineseOrNumber(c rune) bool {
	return unicode.In(c, BertChineseChar, unicode.P) || unicode.IsNumber(c)
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

// SplitPunctuation split punctuation.
func SplitPunctuation(text string) (toks []string) {
	var b strings.Builder
	for _, c := range text {
		if IsPunctuation(c) {
			toks = append(toks, b.String())
			toks = append(toks, string(c))
			b.Reset()
		} else {
			b.WriteRune(c)
		}
	}
	if b.Len() > 0 {
		toks = append(toks, b.String())
	}
	return
}

// BinaryFilter []byte filter space.
func BinaryFilter(arr []byte) []byte {
	result := make([]byte, 0)
	for i := range arr {
		if arr[i] == 0 {
			continue
		}
		if arr[i] != 0 && i != len(arr)-1 && arr[i+1] == 0 {
			if i != 0 {
				result = append(result, byte(' '))
			}
			continue
		}
		result = append(result, arr[i])
	}
	return result
}

// convert functions.
var convertFuncMap = map[string]func([]uint8) interface{}{
	TritonINT32Type: func(b []uint8) interface{} {
		return int32(binary.LittleEndian.Uint32(b))
	},
	TritonINT64Type: func(b []uint8) interface{} {
		return int64(binary.LittleEndian.Uint64(b))
	},
	SliceFloat32Type: func(b []uint8) interface{} {
		return math.Float32frombits(binary.LittleEndian.Uint32(b))
	},
	TritonFP16Type: func(b []uint8) interface{} {
		return math.Float32frombits(binary.LittleEndian.Uint32(b))
	},
	SliceFloat64Type: func(b []uint8) interface{} {
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(b)))
	},
	TritonFP32Type: func(b []uint8) interface{} {
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(b)))
	},
	SliceInt64Type: func(b []uint8) interface{} {
		return int64(binary.LittleEndian.Uint32(b))
	},
	SliceIntType: func(b []uint8) interface{} {
		return int(binary.LittleEndian.Uint32(b))
	},
}

// BinaryToSlice []byte to slice.
func BinaryToSlice(body []uint8, bytesLen int, returnType string) []interface{} {
	// special process BYTES and []byte
	if returnType == TritonBytesType || returnType == SliceByteType {
		return SliceToInterfaceSlice(strings.Fields(string(BinaryFilter(body))))
	}
	// response body split by chunk (other types need convert functions)
	convertFunc := convertFuncMap[returnType]
	convertFuncResult := make([]interface{}, cap(body)/bytesLen)
	for i := 0; i < len(convertFuncResult); i++ {
		if i*bytesLen+bytesLen > len(body) {
			if len(body[i*bytesLen:]) > 0 {
				convertFuncResult[i] = convertFunc(body[i*bytesLen:])
			}
			break
		} else {
			convertFuncResult[i] = convertFunc(body[i*bytesLen : i*bytesLen+bytesLen])
		}
	}
	return convertFuncResult
}
