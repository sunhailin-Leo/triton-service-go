package test

import (
	"encoding/binary"
	"reflect"
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

func TestIsWhiteSpace(t *testing.T) {
	if !utils.IsWhitespace(' ') {
		t.Errorf("function IsWhitespace get error result")
	}
	if utils.IsWhitespace('a') {
		t.Errorf("function IsWhitespace get error result")
	}
}

func TestIsControl(t *testing.T) {
	if utils.IsControl('\t') {
		t.Errorf("function IsControl get error result")
	}
}

func TestIsPunctuation(t *testing.T) {
	tests := []struct {
		input rune
		want  bool
	}{
		{'!', true},
		{',', true},
		{'-', true},
		{'A', false},
		{'1', false},
		{'$', true},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.IsPunctuation(test.input)
		if result != test.want {
			t.Errorf("IsPunctuation(%q) = %v, want %v", test.input, result, test.want)
		}
	}
}

func TestIsChinese(t *testing.T) {
	tests := []struct {
		input rune
		want  bool
	}{
		{'你', true},
		{'好', true},
		{'A', false},
		{'1', false},
		{'$', false},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.IsChinese(test.input)
		if result != test.want {
			t.Errorf("IsChinese(%q) = %v, want %v", test.input, result, test.want)
		}
	}
}

func TestIsChineseOrNumber(t *testing.T) {
	tests := []struct {
		input rune
		want  bool
	}{
		{'你', true},
		{'好', true},
		{'1', true},
		{'A', false},
		{'1', true},
		{'$', false},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.IsChineseOrNumber(test.input)
		if result != test.want {
			t.Errorf("IsChineseOrNumber(%q) = %v, want %v", test.input, result, test.want)
		}
	}
}

func TestIsWhiteSpaceOrChinese(t *testing.T) {
	tests := []struct {
		input rune
		want  bool
	}{
		{'你', true},
		{'好', true},
		{' ', true},
		{'A', false},
		{'1', false},
		{'$', false},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.IsWhiteSpaceOrChinese(test.input)
		if result != test.want {
			t.Errorf("IsWhiteSpaceOrChinese(%q) = %v, want %v", test.input, result, test.want)
		}
	}
}

func TestIsWhiteSpaceOrChineseOrNumber(t *testing.T) {
	tests := []struct {
		input rune
		want  bool
	}{
		{'你', true},
		{'好', true},
		{' ', true},
		{'1', true},
		{'A', false},
		{'$', false},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.IsWhiteSpaceOrChineseOrNumber(test.input)
		if result != test.want {
			t.Errorf("IsWhiteSpaceOrChineseOrNumber(%q) = %v, want %v", test.input, result, test.want)
		}
	}
}

func TestClean(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"Hello\tWorld", "Hello World"},
		{"Invalid\uFFFDCharacter", "InvalidCharacter"},
		{"\x00NullCharacter", "NullCharacter"},
		{"Control\x07Characters\x1B", "ControlCharacters"},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.Clean(test.input)
		if result != test.want {
			t.Errorf("Clean(%q) = %q, want %q", test.input, result, test.want)
		}
	}
}

func TestPadChinese(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"Hello 你好 World", "Hello  你  好  World"},
		{"NoChinese123", "NoChinese123"},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.PadChinese(test.input)
		if result != test.want {
			t.Errorf("PadChinese(%q) = %q, want %q", test.input, result, test.want)
		}
	}
}

func TestCleanAndPadChineseWithWhiteSpace(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"Hello 你好 World", []string{"Hello", "你", "好", "World"}},
		{"中文Chinese", []string{"中", "文", "Chinese"}},
		{"NoChinese123", []string{"NoChinese123"}},
		{"   Multiple   Spaces   ", []string{"Multiple", "Spaces"}},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.CleanAndPadChineseWithWhiteSpace(test.input)
		if !reflect.DeepEqual(result, test.want) {
			t.Errorf("CleanAndPadChineseWithWhiteSpace(%q) = %v, want %v", test.input, result, test.want)
		}
	}
}

func TestStripAccentsAndLower(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"Café", "cafe"},
		{"Résumé", "resume"},
		{"Nǐ hǎo", "ni hao"},
		{"Sören", "soren"},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.StripAccentsAndLower(test.input)
		if result != test.want {
			t.Errorf("StripAccentsAndLower(%q) = %q, want %q", test.input, result, test.want)
		}
	}
}

func TestSplitPunctuation(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"Hello, World!", []string{"Hello", ",", " World", "!"}},
		{"This is a test.", []string{"This is a test", "."}},
		{"NoPunctuation", []string{"NoPunctuation"}},
		{"Comma,semicolon;colon:period.", []string{"Comma", ",", "semicolon", ";", "colon", ":", "period", "."}},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.SplitPunctuation(test.input)
		if !reflect.DeepEqual(result, test.want) {
			t.Errorf("SplitPunctuation(%q) = %v, want %v", test.input, result, test.want)
		}
	}
}

func TestBinaryFilter(t *testing.T) {
	tests := []struct {
		input    []byte
		expected []byte
	}{
		{[]byte{0, 0, 0, 0}, []byte{}},
		{[]byte{1, 2, 3, 4}, []byte{1, 2, 3, 4}},
		// Add more test cases as needed
	}

	for _, test := range tests {
		result := utils.BinaryFilter(test.input)
		if !reflect.DeepEqual(result, test.expected) {
			t.Errorf("BinaryFilter(%v) = %v, expected %v", test.input, result, test.expected)
		}
	}
}

func TestBinaryToSlice_INT32(t *testing.T) {
	// Create binary data for INT32: [1, 2, 3]
	body := make([]byte, 12)
	binary.LittleEndian.PutUint32(body[0:], 1)
	binary.LittleEndian.PutUint32(body[4:], 2)
	binary.LittleEndian.PutUint32(body[8:], 3)

	result := utils.BinaryToSlice(body, 4, utils.TritonINT32Type)
	if len(result) != 3 {
		t.Fatalf("expected 3 elements, got %d", len(result))
	}
	expected := []int32{1, 2, 3}
	for i, exp := range expected {
		if result[i] != exp {
			t.Errorf("at index %d: expected %v, got %v", i, exp, result[i])
		}
	}
}

func TestBinaryToSlice_BYTES(t *testing.T) {
	body := []byte("hello world")
	result := utils.BinaryToSlice(body, 1, utils.TritonBytesType)
	if len(result) != 2 { // "hello" and "world"
		t.Fatalf("expected 2 elements, got %d", len(result))
	}
}
