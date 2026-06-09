package transformers

import (
	"bufio"
	"os"
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
	token    string
	id       ID
	isEnd    bool
}

// newTrieNode creates a new trie node.
func newTrieNode() *trieNode {
	return &trieNode{
		children: make(map[rune]*trieNode),
	}
}

// trie is a prefix tree for efficient token lookup.
type trie struct {
	root *trieNode
}

// newTrie creates a new empty trie.
func newTrie() *trie {
	return &trie{
		root: newTrieNode(),
	}
}

// insert adds a token and its ID to the trie.
func (t *trie) insert(token string, id ID) {
	node := t.root
	for _, ch := range token {
		if _, exists := node.children[ch]; !exists {
			node.children[ch] = newTrieNode()
		}
		node = node.children[ch]
	}
	node.token = token
	node.id = id
	node.isEnd = true
}

// longestPrefix finds the longest matching prefix in the text.
// Returns the matched token, its ID, and the number of bytes consumed from text.
func (t *trie) longestPrefix(text string) (token string, id ID, byteLen int) {
	node := t.root
	var bestToken string
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
			bestToken = node.token
			bestID = node.id
			bestByteLen = endPos
		}
	}

	return bestToken, bestID, bestByteLen
}

// Dict is a container for tokens
// NOTE: python uses an OrderedDict, unsure of implications.
type Dict struct {
	tokens     map[string]ID
	prefixTree *trie
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
	tree := newTrie()
	var idx ID
	for scanner.Scan() {
		token := scanner.Text()
		tokens[token] = idx
		tree.insert(token, idx)
		idx++
	}
	if scanErr := scanner.Err(); scanErr != nil {
		return Dict{}, scanErr
	}
	return Dict{tokens: tokens, prefixTree: tree}, nil
}

// VocabFromSlice will read vocab from config into a Dict.
func VocabFromSlice(vocabArr []string) (Dict, error) {
	if len(vocabArr) == 0 {
		return Dict{}, utils.ErrEmptyVocab
	}
	tokens := make(map[string]ID, len(vocabArr))
	tree := newTrie()
	for i := range vocabArr {
		id := ID(i)
		tokens[vocabArr[i]] = id
		tree.insert(vocabArr[i], id)
	}
	return Dict{tokens: tokens, prefixTree: tree}, nil
}

// New will return a vocab dict from the given tokens, IDs will match index.
func New(tokens []string) Dict {
	v := make(map[string]ID, len(tokens))
	tree := newTrie()
	for i := range tokens {
		id := ID(i)
		v[tokens[i]] = id
		tree.insert(tokens[i], id)
	}
	return Dict{tokens: v, prefixTree: tree}
}

// Add will add an item to the vocabulary, is not thread-safe.
func (v Dict) Add(token string) {
	id := ID(v.Size())
	v.tokens[token] = id
	if v.prefixTree != nil {
		v.prefixTree.insert(token, id)
	}
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

// LongestSubstring returns the longest token in the vocabulary that is a substring of the input.
func (v Dict) LongestSubstring(token string) string {
	if v.prefixTree != nil {
		bestToken := ""
		for i := 0; i < len(token); {
			_, size := utf8.DecodeRuneInString(token[i:])
			if size == 0 {
				break
			}

			matchedToken, _, matchedByteLen := v.prefixTree.longestPrefix(token[i:])
			if matchedByteLen > 0 && len(matchedToken) > len(bestToken) {
				bestToken = matchedToken
			}

			i += size
		}
		return bestToken
	}

	// Fallback to original brute-force method if trie is not available
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
