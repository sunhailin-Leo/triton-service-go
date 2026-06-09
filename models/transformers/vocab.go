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

// trieChild is a key-value pair for trie node children, stored in sorted order.
type trieChild struct {
	key  rune
	node *trieNode
}

// trieNode represents a node in the trie (prefix tree).
// Uses a sorted slice instead of map for children to reduce memory allocations.
type trieNode struct {
	children []trieChild
	id       ID
	isEnd    bool
}

// findChild looks up a child by rune using binary search.
func (n *trieNode) findChild(ch rune) *trieNode {
	lo, hi := 0, len(n.children)
	for lo < hi {
		mid := (lo + hi) >> 1
		if n.children[mid].key < ch {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < len(n.children) && n.children[lo].key == ch {
		return n.children[lo].node
	}
	return nil
}

// addChild inserts a child node in sorted order and returns it.
// Uses the provided arena for allocation if non-nil.
func (n *trieNode) addChild(ch rune, arena *nodeArena) *trieNode {
	// Binary search for insertion point
	lo, hi := 0, len(n.children)
	for lo < hi {
		mid := (lo + hi) >> 1
		if n.children[mid].key < ch {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < len(n.children) && n.children[lo].key == ch {
		return n.children[lo].node
	}
	var child *trieNode
	if arena != nil {
		child = arena.alloc()
	} else {
		child = &trieNode{}
	}
	// Insert at position lo
	n.children = append(n.children, trieChild{})
	copy(n.children[lo+1:], n.children[lo:])
	n.children[lo] = trieChild{key: ch, node: child}
	return child
}

// nodeArena pre-allocates trie nodes in bulk to reduce heap allocations.
type nodeArena struct {
	pool []trieNode
	idx  int
}

// newNodeArena creates an arena with the given capacity.
func newNodeArena(capacity int) *nodeArena {
	return &nodeArena{
		pool: make([]trieNode, capacity),
	}
}

// alloc returns a pointer to the next available node from the arena.
// Falls back to heap allocation if the arena is exhausted.
func (a *nodeArena) alloc() *trieNode {
	if a.idx < len(a.pool) {
		node := &a.pool[a.idx]
		a.idx++
		return node
	}
	return &trieNode{}
}

// trie is a prefix tree for efficient token lookup.
type trie struct {
	root  *trieNode
	arena *nodeArena
}

// newTrie creates a new trie with a pre-allocated node arena.
// estimatedNodes is a hint for the expected total number of trie nodes.
func newTrie(estimatedNodes int) *trie {
	arena := newNodeArena(estimatedNodes)
	root := arena.alloc()
	return &trie{
		root:  root,
		arena: arena,
	}
}

// insert adds a token and its ID to the trie.
func (t *trie) insert(token string, id ID) {
	node := t.root
	for _, ch := range token {
		node = node.addChild(ch, t.arena)
	}
	node.id = id
	node.isEnd = true
}

// longestPrefix finds the longest matching prefix in the text.
// Returns the matched token's ID and the number of bytes consumed from text.
// The caller can recover the matched string via text[:byteLen].
func (t *trie) longestPrefix(text string) (id ID, byteLen int) {
	node := t.root
	var bestID ID
	var bestByteLen int

	for i, ch := range text {
		child := node.findChild(ch)
		if child == nil {
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
	// Pre-allocate arena: ~2 nodes per token accounts for prefix sharing in typical vocabularies.
	tree := newTrie(estimatedSize * 2)
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
	tree := newTrie(len(vocabArr) * 2)
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
	tree := newTrie(len(tokens) * 2)
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

			_, matchedByteLen := v.prefixTree.longestPrefix(token[i:])
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
