package transformers

import (
	"bufio"
	"os"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

// Provider is an interface for exposing a vocab.
type Provider interface {
	Vocab() Dict
}

// ID is used to identify vocab items.
type ID int32

// Int64 int32 ID to int64.
func (id ID) Int64() int64 {
	return int64(id)
}

// Dict is a container for tokens
// NOTE: python uses an OrderedDict, unsure of implications.
type Dict struct {
	tokens map[string]ID
}

// VocabFromFile will read a newline delimited file into a Dict.
func VocabFromFile(path string) (Dict, error) {
	f, err := os.Open(path)
	if err != nil {
		return Dict{}, err
	}
	defer func() { _ = f.Close() }()

	// Estimate vocab size from file size (average ~10 bytes per token for BERT vocab)
	fi, statErr := f.Stat()
	estimatedSize := 21128 // default BERT vocab size
	if statErr == nil && fi.Size() > 0 {
		estimatedSize = int(fi.Size() / 10)
		if estimatedSize < 100 {
			estimatedSize = 100
		}
	}

	scanner := bufio.NewScanner(f)
	voc := Dict{tokens: make(map[string]ID, estimatedSize)}
	for scanner.Scan() {
		voc.Add(scanner.Text())
	}
	if scanErr := scanner.Err(); scanErr != nil {
		return Dict{}, scanErr
	}
	return voc, nil
}

// VocabFromSlice will read vocab from config into a Dict.
func VocabFromSlice(vocabArr []string) (Dict, error) {
	if len(vocabArr) == 0 {
		return Dict{}, utils.ErrEmptyVocab
	}
	voc := Dict{tokens: make(map[string]ID, len(vocabArr))}
	for i := range vocabArr {
		voc.Add(vocabArr[i])
	}
	return voc, nil
}

// New will return a vocab dict from the given tokens, IDs will match index.
func New(tokens []string) Dict {
	v := make(map[string]ID, len(tokens))
	for i := range tokens {
		v[tokens[i]] = ID(i)
	}
	return Dict{tokens: v}
}

// Add will add an item to the vocabulary, is not thread-safe.
func (v Dict) Add(token string) {
	v.tokens[token] = ID(v.Size())
}

// GetID will return the ID of the token in the vocab. Will be negative if it doesn't exist.
func (v Dict) GetID(token string) ID {
	id, ok := v.tokens[token]
	if !ok {
		return ID(-1)
	}
	return id
}

// Size returns the size of the vocabulary.
func (v Dict) Size() int {
	return len(v.tokens)
}

// LongestSubstring returns the longest token that is a substring of the token.
func (v Dict) LongestSubstring(token string) string {
	// Greedy, optimize to trie if needed
	for i := len(token); i > 0; i-- {
		sub := token[:i]
		if v.IsInVocab(sub) {
			return sub
		}
	}
	return ""
}

// ConvertItems convert items to ids.
func (v Dict) ConvertItems(items []string) []ID {
	ids := make([]ID, len(items))
	for i := range items {
		ids[i] = v.tokens[items[i]]
	}
	return ids
}

// ConvertTokens convert token to id.
func (v Dict) ConvertTokens(tokens []string) []ID {
	return v.ConvertItems(tokens)
}

// IsInVocab token is in vocabs.
func (v Dict) IsInVocab(token string) bool {
	_, exists := v.tokens[token]

	return exists
}
