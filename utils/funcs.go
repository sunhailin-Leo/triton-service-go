package utils

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// IsWhitespace checks whether rune c is a BERT whitespace character
func IsWhitespace(c rune) bool {
	switch c {
	case ' ':
		return true
	case '\t':
		return true
	case '\n':
		return true
	case '\r':
		return true
	}
	return unicode.Is(unicode.Zs, c)
}

// IsControl checks whether rune c is a BERT control character
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

// IsPunctuation checks whether rune c is a BERT punctuation character
func IsPunctuation(c rune) bool {
	//return unicode.In(c, utils.Bp, unicode.P)
	//return unicode.In(c, utils.AsciiPunctuation, unicode.P) && c != '-'
	return unicode.In(c, AsciiPunctuation, unicode.P)
}

// IsChinese validates that rune c is in the CJK range according to BERT spec
func IsChinese(c rune) bool {
	// unicode.Is(unicode.Han, c)
	return unicode.In(c, BertChineseChar, unicode.P)
}

// IsWhiteSpaceOrChinese validates that rune c is whitespace or is Chinese
func IsWhiteSpaceOrChinese(c rune) bool {
	return IsWhitespace(c) || IsChinese(c)
}

// Clean function will clear some characters
func Clean(text string) string {
	var b strings.Builder
	for _, c := range text {
		if c == 0 || c == 0xfffd || IsControl(c) {
			continue
		} else if IsWhitespace(c) {
			b.WriteRune(' ')
		} else {
			b.WriteRune(c)
		}
	}
	return b.String()
}

// PadChinese will add space padding around all CJK chars
// This implementation matches BasicTokenizer._tokenize_chinese_chars
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

// CleanAndPadChineseWithWhiteSpace combine three function clean, padChinese, tokenizeWhitespaceV1
func CleanAndPadChineseWithWhiteSpace(text string) []string {
	var b strings.Builder
	for _, c := range text {
		if c == 0 || c == 0xfffd || IsControl(c) {
			continue
		} else if IsChinese(c) {
			b.WriteRune(' ')
			b.WriteRune(c)
			b.WriteRune(' ')
		} else if IsWhitespace(c) {
			b.WriteRune(' ')
		} else {
			b.WriteRune(c)
		}
	}
	return strings.Fields(strings.TrimSpace(b.String()))
}

// StripAccentsAndLower stripAccentsAndLower
func StripAccentsAndLower(text string) string {
	var b strings.Builder
	for _, c := range norm.NFD.String(text) {
		if unicode.Is(unicode.Mn, c) {
			continue
		}
		b.WriteRune(unicode.ToLower(c))
		//if !unicode.Is(unicode.Mn, c) {
		//	b.WriteRune(unicode.ToLower(c))
		//}
	}
	return b.String()
}

// SplitPunctuation splitPunctuation
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

// AsciiPunctuation Ascii punctuation characters range
var AsciiPunctuation = &unicode.RangeTable{
	R16: []unicode.Range16{
		{0x0021, 0x002f, 1}, // 33-47
		{0x003a, 0x0040, 1}, // 58-64
		{0x005b, 0x0060, 1}, // 91-96
		{0x007b, 0x007e, 1}, // 123-126
	},
	LatinOffset: 4, // All less than 0x00FF
}

// BertChineseChar maybe is the BERT Chinese Char...
var BertChineseChar = &unicode.RangeTable{
	R16: []unicode.Range16{
		{0x4e00, 0x9fff, 1},
		{0x3400, 0x4dbf, 1},
		{0xf900, 0xfaff, 1},
	},
	R32: []unicode.Range32{
		{Lo: 0x20000, Hi: 0x2a6df, Stride: 1},
		{Lo: 0x2a700, Hi: 0x2b73f, Stride: 1},
		{Lo: 0x2b740, Hi: 0x2b81f, Stride: 1},
		{Lo: 0x2b820, Hi: 0x2ceaf, Stride: 1},
		{Lo: 0x2f800, Hi: 0x2fa1f, Stride: 1},
	},
}

// StringSliceTruncate truncate uses heuristic of trimming seq with longest len until sequenceLen satisfied
func StringSliceTruncate(sequence [][]string, maxLen int) [][]string {
	for sequenceLen := len(sequence[0]); sequenceLen > maxLen; sequenceLen-- {
		// Sort to get the longest first
		var mi, mv int
		for i := len(sequence) - 1; i >= 0; i-- {
			// iterate in reverse to select lower indexes
			seq := sequence[i]
			if len(seq) > mv {
				mi = i
				mv = len(seq)
			}
		}
		// can't trim anymore
		if mv <= 0 {
			return sequence
		}
		rm := sequence[mi]
		// Mark for GC, avoid mem leak
		rm[len(rm)-1] = ""
		sequence[mi] = rm[:len(rm)-1]
	}
	return sequence
}

// SliceTransposeFor3D Transport 3-D Dimension Slice. Like NxM to MxN
func SliceTransposeFor3D[T comparable](slice [][][]T) [][][]T {
	n, m := len(slice), len(slice[0])
	transposed := make([][][]T, m)
	for i := range transposed {
		transposed[i] = make([][]T, n)
	}
	// transposed
	for i := range slice {
		for j := range slice[i] {
			transposed[j][i] = slice[i][j]
		}
	}
	return transposed
}

// SliceTransposeFor2D Transport 2-D Dimension Slice. Like NxM to MxN
func SliceTransposeFor2D[T comparable](slice [][]T) [][]T {
	n, m := len(slice), len(slice[0])
	transposed := make([][]T, m)
	for i := range transposed {
		transposed[i] = make([]T, n)
	}
	// transposed
	for i := range slice {
		for j := range slice[i] {
			transposed[j][i] = slice[i][j]
		}
	}
	return transposed
}
