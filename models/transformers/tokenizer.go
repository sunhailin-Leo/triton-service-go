package transformers

import (
	"strings"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

const (
	DefaultCLS          string = "[CLS]"
	DefaultSEP          string = "[SEP]"
	DefaultUNK          string = "[UNK]"
	DefaultMask         string = "[MASK]"
	NumPadToken         string = "##"
	DefaultMaxWordChars int    = 200
	DataSplitString     string = " ||| "
)

// TokenizerV1 is implemented by any value that has the Tokenize method.
type TokenizerV1 interface {
	Tokenize(text string) []StringOffsetsPair
	TokenizeChinese(text string) []StringOffsetsPair
	TokenizeChineseCharMode(text string) []StringOffsetsPair
}

// StringOffsetsPair represents a string value paired with offsets bounds.
// It usually represents a token string and its offsets positions in the
// original string.
type StringOffsetsPair struct {
	String  string
	Offsets OffsetsType
}

// OffsetsType represents a (start, end) offsets pair.
// It usually represents a lower inclusive index position, and an upper
// exclusive position.
type OffsetsType struct {
	Start int
	End   int
}

// GetStrings returns a sequence of string values from the given slice
// of StringOffsetsPair.
func GetStrings(tokens []StringOffsetsPair) []string {
	result := make([]string, len(tokens))
	for i := range tokens {
		result[i] = tokens[i].String
	}
	return result
}

// GetOffsets returns a sequence of offsets values from the given slice
// of StringOffsetsPair.
func GetOffsets(tokens []StringOffsetsPair) []OffsetsType {
	result := make([]OffsetsType, len(tokens))
	for i := range tokens {
		result[i] = tokens[i].Offsets
	}
	return result
}

// BaseTokenizer is a straightforward tokenizer implementations, which
// splits by whitespace and punctuation characters.
type BaseTokenizer struct {
	specialWords map[string]struct{}
}

// OptionV1 allows to configure a new BaseTokenizer with your specific needs.
type OptionV1 func(*BaseTokenizer)

// RegisterSpecialWords is an option to register a special word.
func RegisterSpecialWords(specialWords ...string) OptionV1 {
	return func(f *BaseTokenizer) {
		for i := range specialWords {
			f.specialWords[specialWords[i]] = struct{}{}
		}
	}
}

// NewBaseTokenizer returns a new base tokenizer ready to use.
func NewBaseTokenizer(opts ...OptionV1) *BaseTokenizer {
	t := &BaseTokenizer{
		specialWords: make(map[string]struct{}),
	}
	for i := range opts {
		opts[i](t)
	}
	return t
}

// Tokenize converts the input text to a slice of tokens, where each token is a white-separated word,
// a number or a punctuation sign.
// The resulting tokens preserve the alignment with the portion of the original text they belong to.
func (t *BaseTokenizer) Tokenize(text string) []StringOffsetsPair {
	splitTokens := make([]StringOffsetsPair, 0)
	text = utils.Clean(text)
	spaceTokens := t.splitOn(text, utils.IsWhitespace, false)

	for i := range spaceTokens {
		if _, isSpecial := t.specialWords[spaceTokens[i].String]; isSpecial {
			splitTokens = append(splitTokens, spaceTokens[i])
			continue // TODO: this is temporary solution to don't split special tokens further; improve it.
		}

		puncTokens := t.splitOn(spaceTokens[i].String, utils.IsPunctuation, true)
		for j := range puncTokens {
			splitTokens = append(splitTokens, StringOffsetsPair{
				String: puncTokens[j].String,
				Offsets: OffsetsType{
					Start: spaceTokens[i].Offsets.Start + puncTokens[j].Offsets.Start,
					End:   spaceTokens[i].Offsets.Start + puncTokens[j].Offsets.End,
				},
			})
		}
	}
	return splitTokens
}

// TokenizeChinese Like Tokenize but focus on Chinese.
func (t *BaseTokenizer) TokenizeChinese(text string) []StringOffsetsPair {
	splitTokens := make([]StringOffsetsPair, 0)
	spaceTokens := t.splitOnChinese(text, utils.IsWhiteSpaceOrChinese, false, utils.IsChinese)

	for i := range spaceTokens {
		if _, isSpecial := t.specialWords[spaceTokens[i].String]; isSpecial {
			splitTokens = append(splitTokens, spaceTokens[i])
			continue // TODO: this is temporary solution to don't split special tokens further; improve it.
		}

		puncTokens := t.splitOnChinese(
			utils.StripAccentsAndLower(spaceTokens[i].String), utils.IsPunctuation, true, utils.IsChinese)
		for j := range puncTokens {
			splitTokens = append(splitTokens, StringOffsetsPair{
				String: puncTokens[j].String,
				Offsets: OffsetsType{
					Start: spaceTokens[i].Offsets.Start + puncTokens[j].Offsets.Start,
					End:   spaceTokens[i].Offsets.Start + puncTokens[j].Offsets.End,
				},
			})
		}
	}
	return splitTokens
}

// TokenizeChineseCharMode Like TokenizeChinese but focus on Chinese NER.
func (t *BaseTokenizer) TokenizeChineseCharMode(text string) []StringOffsetsPair {
	splitTokens := make([]StringOffsetsPair, 0)
	spaceTokens := t.splitOnChinese(text, utils.IsWhiteSpaceOrChineseOrNumber, false, utils.IsChineseOrNumber)

	for i := range spaceTokens {
		if _, isSpecial := t.specialWords[spaceTokens[i].String]; isSpecial {
			splitTokens = append(splitTokens, spaceTokens[i])
			continue // TODO: this is temporary solution to don't split special tokens further; improve it.
		}

		puncTokens := t.splitOnChinese(
			utils.StripAccentsAndLower(spaceTokens[i].String), utils.IsPunctuation, true, utils.IsChineseOrNumber)
		for j := range puncTokens {
			splitTokens = append(splitTokens, StringOffsetsPair{
				String: puncTokens[j].String,
				Offsets: OffsetsType{
					Start: spaceTokens[i].Offsets.Start + puncTokens[j].Offsets.Start,
					End:   spaceTokens[i].Offsets.Start + puncTokens[j].Offsets.End,
				},
			})
		}
	}
	return splitTokens
}

// splitOn splits the given string as the `shouldSplit` predicate dictates.
// It keeps track of the offsets.
func (t *BaseTokenizer) splitOn(text string, shouldSplit func(rune) bool, includeSplitToken bool) []StringOffsetsPair {
	words := make([]StringOffsetsPair, 0)
	var word []rune

	offset := 0
	for _, r := range text {
		if shouldSplit(r) {
			wordLen := len(word)
			if wordLen > 0 {
				words = append(words, StringOffsetsPair{
					String:  string(word),
					Offsets: OffsetsType{Start: offset - wordLen, End: offset},
				})
				word = make([]rune, 0, cap(word))
			}
			if includeSplitToken {
				words = append(words, StringOffsetsPair{
					String:  string(r),
					Offsets: OffsetsType{Start: offset, End: offset + 1},
				})
			}
		} else {
			word = append(word, r)
		}
		offset++
	}

	// Don't forget the potential last word
	if len(word) > 0 {
		words = append(words, StringOffsetsPair{
			String:  string(word),
			Offsets: OffsetsType{Start: offset - len(word), End: offset},
		})
	}
	return words
}

// splitOnChinese splits the given string as the `shouldSplit` predicate dictates.
// It keeps track of the offsets.
func (t *BaseTokenizer) splitOnChinese(text string, shouldSplit func(rune) bool, includeSplitToken bool, includeSplitFunc func(rune) bool) []StringOffsetsPair {
	words := make([]StringOffsetsPair, 0)
	var word []rune

	offset := 0
	for _, r := range text {
		if shouldSplit(r) {
			wordLen := len(word)
			if wordLen > 0 {
				words = append(words, StringOffsetsPair{
					String:  string(word),
					Offsets: OffsetsType{Start: offset - wordLen, End: offset},
				})
				word = make([]rune, 0, cap(word))
			}
			if includeSplitToken || includeSplitFunc(r) {
				words = append(words, StringOffsetsPair{
					String:  string(r),
					Offsets: OffsetsType{Start: offset, End: offset + 1},
				})
			}
		} else {
			word = append(word, r)
		}
		offset++
	}

	// Don't forget the potential last word
	if len(word) > 0 {
		words = append(words, StringOffsetsPair{
			String:  string(word),
			Offsets: OffsetsType{Start: offset - len(word), End: offset},
		})
	}
	return words
}

// WordPieceTokenizer is a tokenizer that breaks tokens into sub-word units based on a supplied vocabulary.
// See https://arxiv.org/pdf/1609.08144.pdf Section 4.1 for details.
// WordPieceTokenizers uses BaseTokenizer to preprocess the input text.
type WordPieceTokenizer struct {
	baseTokenizer *BaseTokenizer
	vocabulary    Dict
	unkToken      string
	splitPrefix   string
	maxWordChars  int
	neverSplit    []string
}

// NewWordPieceTokenizer returns a new WordPieceTokenizer.
func NewWordPieceTokenizer(vocabulary Dict) *WordPieceTokenizer {
	return &WordPieceTokenizer{
		baseTokenizer: NewBaseTokenizer(RegisterSpecialWords(DefaultUNK, DefaultCLS, DefaultSEP, DefaultMask)),
		vocabulary:    vocabulary,
		unkToken:      DefaultUNK,
		splitPrefix:   NumPadToken,
		maxWordChars:  DefaultMaxWordChars,
		neverSplit:    []string{DefaultCLS, DefaultSEP, DefaultUNK, DefaultMask},
	}
}

// Tokenize converts the input text to a slice of words or sub-words token units based on the supplied vocabulary.
// The resulting tokens preserve the alignment with the portion of the original text they belong to.
func (t *WordPieceTokenizer) Tokenize(text string) []StringOffsetsPair {
	return t.WordPieceTokenize(t.baseTokenizer.Tokenize(text))
}

// TokenizeChinese Like Tokenize but focus on Chinese
func (t *WordPieceTokenizer) TokenizeChinese(text string) []StringOffsetsPair {
	return t.WordPieceTokenize(t.baseTokenizer.TokenizeChinese(text))
}

// TokenizeChineseCharMode Like TokenizeChinese but focus on Chinese NER
func (t *WordPieceTokenizer) TokenizeChineseCharMode(text string) []StringOffsetsPair {
	return t.WordPieceTokenize(t.baseTokenizer.TokenizeChineseCharMode(text))
}

// WordPieceTokenize
// transforms the input token in a new slice of words or sub-words units based on the supplied vocabulary.
// The resulting tokens preserve the alignment with the portion of the original text they belong to.
func (t *WordPieceTokenizer) WordPieceTokenize(tokens []StringOffsetsPair) []StringOffsetsPair {
	outputTokens := make([]StringOffsetsPair, 0)
	for i := range tokens {
		characters := []rune(tokens[i].String)

		if len(characters) > t.maxWordChars {
			if t.vocabulary.GetID(t.unkToken) == -1 {
				panic("Missing unk-token")
			}
			outputTokens = append(outputTokens, StringOffsetsPair{String: t.unkToken, Offsets: tokens[i].Offsets})
			continue
		}

		isBad := false
		start := 0
		subTokens := make([]StringOffsetsPair, 0)

		for start < len(characters) {
			end := len(characters)
			var curStrToken StringOffsetsPair
			found := false

			for start < end {
				subStr := string(characters[start:end])
				if start > 0 {
					subStr = t.splitPrefix + subStr
				}
				if t.vocabulary.GetID(subStr) != -1 {
					found = true
					curStrToken.String = subStr
					curStrToken.Offsets = OffsetsType{
						Start: tokens[i].Offsets.Start + start,
						End:   tokens[i].Offsets.Start + end,
					}
					break
				}
				end--
			}
			if !found {
				isBad = true

				break
			}
			subTokens = append(subTokens, curStrToken)
			start = end
		}

		if isBad {
			if t.vocabulary.GetID(t.unkToken) == -1 {
				panic("Missing unk-token")
			}
			outputTokens = append(outputTokens, StringOffsetsPair{String: t.unkToken, Offsets: tokens[i].Offsets})
		} else {
			outputTokens = append(outputTokens, subTokens...)
		}
	}
	return outputTokens
}

// IsDefaultSpecial return whether the word matches a special token, or not.
func IsDefaultSpecial(word string) bool {
	switch word {
	case DefaultUNK, DefaultCLS, DefaultSEP, DefaultMask:
		return true
	default:
		return false
	}
}

// TokensRange represents an index offsets pair of a token.
type TokensRange struct {
	Start int
	End   int
}

// GroupPieces returns a list of tokens range each of which represents
// the start and the end index of the tokens that form a complete word.
func GroupPieces(tokens []StringOffsetsPair) []TokensRange {
	groups := make([]TokensRange, 0)
	for i := range tokens {
		if strings.HasPrefix(tokens[i].String, NumPadToken) {
			groups[len(groups)-1].End = i
		} else {
			groups = append(groups, TokensRange{Start: i, End: i})
		}
	}
	return groups
}

// MakeOffsetPairsFromGroups creates a sequence tokenizers.StringOffsetsPair
// elements from the given groups.
func MakeOffsetPairsFromGroups(text string, tokens []StringOffsetsPair, groups []TokensRange) []StringOffsetsPair {
	outputTokens := make([]StringOffsetsPair, len(groups))
	for i := range groups {
		startToken, endToken := tokens[groups[i].Start], tokens[groups[i].End]
		outputTokens[i] = StringOffsetsPair{
			String:  string([]rune(text)[startToken.Offsets.Start:endToken.Offsets.End]),
			Offsets: OffsetsType{Start: startToken.Offsets.Start, End: endToken.Offsets.End},
		}
	}
	return outputTokens
}
