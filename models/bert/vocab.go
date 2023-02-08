package bert

import (
	"bufio"
	"errors"
	"os"
)

// Provider is an interface for exposing a vocab
type Provider interface {
	Vocab() Dict
}

// ID is used to identify vocab items
type ID int32

// Int64 int32 ID to int64
func (id ID) Int64() int64 {
	return int64(id)
}

// Dict is a container for tokens
// NOTE: python uses an OrderedDict, unsure of implications
type Dict struct {
	tokens map[string]ID
}

// VocabFromFile will read a newline delimited file into a Dict
func VocabFromFile(path string) (Dict, error) {
	f, err := os.Open(path)
	if err != nil {
		// TODO wrap w/ stdlib
		return Dict{}, err
	}
	defer func() { _ = f.Close() }()
	scanner := bufio.NewScanner(f)
	voc := Dict{tokens: map[string]ID{}}
	for scanner.Scan() {
		voc.Add(scanner.Text())
	}
	return voc, nil
}

// VocabFromSlice will read vocab from config into a Dict
func VocabFromSlice(vocabArr []string) (Dict, error) {
	if len(vocabArr) == 0 {
		return Dict{}, errors.New("empty vocab")
	}
	voc := Dict{tokens: map[string]ID{}}
	for _, vocab := range vocabArr {
		voc.Add(vocab)
	}
	return voc, nil
}

// New will return a vocab dict from the given tokens, IDs will match index
func New(tokens []string) Dict {
	v := make(map[string]ID, len(tokens))
	for i, t := range tokens {
		v[t] = ID(i)
	}
	return Dict{tokens: v}
}

// Add will add an item to the vocabulary, is not thread-safe
func (v Dict) Add(token string) {
	v.tokens[token] = ID(v.Size())
}

// GetID will return the ID of the token in the vocab. Will be negative if it doesn't exist
func (v Dict) GetID(token string) ID {
	id, ok := v.tokens[token]
	if !ok {
		return ID(-1)
	}
	return id
}

// Size returns the size of the vocabulary
func (v Dict) Size() int {
	return len(v.tokens)
}

// LongestSubstring returns the longest token that is a substring of the token
func (v Dict) LongestSubstring(token string) string {
	// Greedt, optimize to trie if needed
	for i := len(token); i > 0; i-- {
		sub := token[:i]
		if v.IsInVocab(sub) {
			return sub
		}
	}
	return ""
}

// ConvertItems convert items to ids
func (v Dict) ConvertItems(items []string) []ID {
	ids := make([]ID, len(items))
	for i, m := range items {
		ids[i] = v.tokens[m]
	}
	return ids
}

// ConvertTokens convert token to id
func (v Dict) ConvertTokens(tokens []string) []ID {
	return v.ConvertItems(tokens)
}

func (v Dict) IsInVocab(token string) bool {
	_, exists := v.tokens[token]
	return exists
}
