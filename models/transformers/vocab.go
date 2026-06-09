package transformers

import (
	"bufio"
	"os"
	"sync"
	"unicode/utf8"

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

// trieNode represents a node in the trie (prefix tree).
type trieNode struct {
	children map[rune]*trieNode
	id       ID
	isEnd    bool
}

// trie is a prefix tree for efficient longest-prefix token lookup.
type trie struct {
	root *trieNode
}

// buildTrie constructs a trie from the given token map.
func buildTrie(tokens map[string]ID) *trie {
	t := &trie{root: &trieNode{children: make(map[rune]*trieNode)}}
	for token, id := range tokens {
		node := t.root
		for _, ch := range token {
			child, exists := node.children[ch]
			if !exists {
				child = &trieNode{children: make(map[rune]*trieNode)}
				node.children[ch] = child
			}
			node = child
		}
		node.id = id
		node.isEnd = true
	}
	return t
}

// longestPrefix finds the longest matching prefix in the text.
// Returns the matched token's ID and the number of bytes consumed from text.
// The caller can recover the matched string via text[:byteLen].
func (t *trie) longestPrefix(text string) (id ID, byteLen int) {
	node := t.root
	var bestID ID
	var bestByteLen int

	for i, ch := range text {
		child, exists := node.children[ch]
		if !exists {
			break
		}
		node = child
		// i is the byte offset of ch; add utf8.RuneLen(ch) to get the end byte position
		endPos := i + utf8.RuneLen(ch)
		if node.isEnd {
			bestID = node.id
			bestByteLen = endPos
		}
	}

	return bestID, bestByteLen
}

// Dict is a container for tokens
// NOTE: python uses an OrderedDict, unsure of implications.
type Dict struct {
	tokens   map[string]ID
	trieOnce sync.Once
	trie     *trie
}

// ensureTrie lazily builds the trie on first use.
func (v *Dict) ensureTrie() *trie {
	v.trieOnce.Do(func() {
		v.trie = buildTrie(v.tokens)
	})
	return v.trie
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
	tokens := make(map[string]ID, estimatedSize)
	var idx ID
	for scanner.Scan() {
		tokens[scanner.Text()] = idx
		idx++
	}
	if scanErr := scanner.Err(); scanErr != nil {
		return Dict{}, scanErr
	}
	return Dict{tokens: tokens}, nil
}

// VocabFromSlice will read vocab from config into a Dict.
func VocabFromSlice(vocabArr []string) (Dict, error) {
	if len(vocabArr) == 0 {
		return Dict{}, utils.ErrEmptyVocab
	}
	tokens := make(map[string]ID, len(vocabArr))
	for i := range vocabArr {
		tokens[vocabArr[i]] = ID(i)
	}
	return Dict{tokens: tokens}, nil
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
// Note: invalidates the lazily-built trie; it will be rebuilt on next use.
func (v *Dict) Add(token string) {
	id := ID(v.Size())
	v.tokens[token] = id
	// Reset trie so it gets rebuilt with the new token on next use.
	v.trie = nil
	v.trieOnce = sync.Once{}
}

// GetID will return the ID of the token in the vocab. Will be negative if it doesn't exist.
func (v *Dict) GetID(token string) ID {
	id, ok := v.tokens[token]
	if !ok {
		return ID(-1)
	}
	return id
}

// Size returns the size of the vocabulary.
func (v *Dict) Size() int {
	return len(v.tokens)
}

// LongestSubstring returns the longest token in the vocabulary that is a substring of the input.
func (v *Dict) LongestSubstring(token string) string {
	tree := v.ensureTrie()
	bestToken := ""
	for i := 0; i < len(token); {
		_, size := utf8.DecodeRuneInString(token[i:])
		if size == 0 {
			break
		}

		_, matchedByteLen := tree.longestPrefix(token[i:])
		if matchedByteLen > 0 {
			matched := token[i : i+matchedByteLen]
			if len(matched) > len(bestToken) {
				bestToken = matched
			}
		}

		i += size
	}
	return bestToken
}

// ConvertItems convert items to ids.
func (v *Dict) ConvertItems(items []string) []ID {
	ids := make([]ID, len(items))
	for i := range items {
		ids[i] = v.tokens[items[i]]
	}
	return ids
}

// ConvertTokens convert token to id.
func (v *Dict) ConvertTokens(tokens []string) []ID {
	return v.ConvertItems(tokens)
}

// IsInVocab token is in vocabs.
func (v *Dict) IsInVocab(token string) bool {
	_, exists := v.tokens[token]

	return exists
}
