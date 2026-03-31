package utils_test

import (
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

func TestPadSlice(t *testing.T) {
	tests := []struct {
		name         string
		slice        []int
		padLength    int
		paddingValue int
		expected     []int
	}{
		{
			name:         "normal padding",
			slice:        []int{1, 2, 3, 4},
			padLength:    6,
			paddingValue: 0,
			expected:     []int{1, 2, 3, 4, 0, 0},
		},
		{
			name:         "already full",
			slice:        []int{1, 2, 3, 4},
			padLength:    4,
			paddingValue: 0,
			expected:     []int{1, 2, 3, 4},
		},
		{
			name:         "longer than pad length",
			slice:        []int{1, 2, 3, 4, 5},
			padLength:    4,
			paddingValue: 0,
			expected:     []int{1, 2, 3, 4, 5},
		},
		{
			name:         "empty slice padding",
			slice:        []int{},
			padLength:    3,
			paddingValue: 99,
			expected:     []int{99, 99, 99},
		},
		{
			name:         "no padding needed",
			slice:        []int{1, 2},
			padLength:    2,
			paddingValue: 0,
			expected:     []int{1, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.PadSlice(tt.slice, tt.padLength, tt.paddingValue)
			if len(result) != len(tt.expected) {
				t.Errorf("PadSlice() = %v, want %v", result, tt.expected)
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("PadSlice()[%d] = %v, want %v", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestStringSliceTruncate(t *testing.T) {
	tests := []struct {
		name     string
		sequence [][]string
		maxLen   int
		expected [][]string
	}{
		{
			name:     "normal truncate",
			sequence: [][]string{{"a", "b", "c"}, {"d", "e", "f"}},
			maxLen:   2,
			expected: [][]string{{"a", "b", "c"}, {"d", "e"}},
		},
		{
			name:     "no truncate needed",
			sequence: [][]string{{"a", "b"}, {"c", "d"}},
			maxLen:   3,
			expected: [][]string{{"a", "b"}, {"c", "d"}},
		},
		{
			name:     "empty elements",
			sequence: [][]string{{"a", "b", "c"}, {"d"}},
			maxLen:   2,
			expected: [][]string{{"a", "b"}, {"d"}},
		},
		{
			name:     "truncate to zero",
			sequence: [][]string{{"a", "b", "c"}, {"d", "e", "f"}},
			maxLen:   0,
			expected: [][]string{{"a", "b"}, {"d"}},
		},
		{
			name:     "uneven lengths",
			sequence: [][]string{{"a", "b", "c", "d"}, {"e", "f"}},
			maxLen:   2,
			expected: [][]string{{"a", "b"}, {"e", "f"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.StringSliceTruncate(tt.sequence, tt.maxLen)
			if len(result) != len(tt.expected) {
				t.Errorf("StringSliceTruncate() = %v, want %v", result, tt.expected)
			}
			for i := range result {
				if len(result[i]) != len(tt.expected[i]) {
					t.Errorf("StringSliceTruncate()[%d] = %v, want %v", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestSliceTransposeFor3D(t *testing.T) {
	tests := []struct {
		name     string
		slice    [][][]int
		expected [][][]int
	}{
		{
			name: "normal transpose",
			slice: [][][]int{
				{{1, 2}, {3, 4}},
				{{5, 6}, {7, 8}},
			},
			expected: [][][]int{
				{{1, 2}, {5, 6}},
				{{3, 4}, {7, 8}},
			},
		},
		{
			name: "3x3 transpose",
			slice: [][][]int{
				{{1, 2, 3}, {4, 5, 6}},
				{{7, 8, 9}, {10, 11, 12}},
				{{13, 14, 15}, {16, 17, 18}},
			},
			expected: [][][]int{
				{{1, 2, 3}, {7, 8, 9}, {13, 14, 15}},
				{{4, 5, 6}, {10, 11, 12}, {16, 17, 18}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.SliceTransposeFor3D(tt.slice)
			if len(result) != len(tt.expected) {
				t.Errorf("SliceTransposeFor3D() length mismatch: got %d, want %d", len(result), len(tt.expected))
			}
			for i := range result {
				for j := range result[i] {
					for k := range result[i][j] {
						if result[i][j][k] != tt.expected[i][j][k] {
							t.Errorf("SliceTransposeFor3D()[%d][%d][%d] = %d, want %d", i, j, k, result[i][j][k], tt.expected[i][j][k])
						}
					}
				}
			}
		})
	}
}

func TestSliceTransposeFor2D(t *testing.T) {
	tests := []struct {
		name     string
		slice    [][]int
		expected [][]int
	}{
		{
			name:     "normal transpose",
			slice:    [][]int{{1, 2, 3}, {4, 5, 6}},
			expected: [][]int{{1, 4}, {2, 5}, {3, 6}},
		},
		{
			name:     "square transpose",
			slice:    [][]int{{1, 2}, {3, 4}},
			expected: [][]int{{1, 3}, {2, 4}},
		},
		{
			name:     "single row",
			slice:    [][]int{{1, 2, 3}},
			expected: [][]int{{1}, {2}, {3}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.SliceTransposeFor2D(tt.slice)
			if len(result) != len(tt.expected) {
				t.Errorf("SliceTransposeFor2D() length mismatch: got %d, want %d", len(result), len(tt.expected))
			}
			for i := range result {
				for j := range result[i] {
					if result[i][j] != tt.expected[i][j] {
						t.Errorf("SliceTransposeFor2D()[%d][%d] = %d, want %d", i, j, result[i][j], tt.expected[i][j])
					}
				}
			}
		})
	}
}

func TestFlatten2DSlice(t *testing.T) {
	tests := []struct {
		name     string
		arr      [][]int
		expected []int
	}{
		{
			name:     "normal flatten",
			arr:      [][]int{{1, 2}, {3, 4}, {5, 6}},
			expected: []int{1, 2, 3, 4, 5, 6},
		},
		{
			name:     "empty slice",
			arr:      [][]int{},
			expected: []int{},
		},
		{
			name:     "single element",
			arr:      [][]int{{1}},
			expected: []int{1},
		},
		{
			name:     "uneven rows",
			arr:      [][]int{{1, 2}, {3}, {4, 5, 6}},
			expected: []int{1, 2, 3, 4, 5, 6},
		},
		{
			name:     "empty rows",
			arr:      [][]int{{}, {1, 2}, {}},
			expected: []int{1, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.Flatten2DSlice(tt.arr)
			if len(result) != len(tt.expected) {
				t.Errorf("Flatten2DSlice() = %v, want %v", result, tt.expected)
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("Flatten2DSlice()[%d] = %d, want %d", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestGetMaxSubSliceLength(t *testing.T) {
	tests := []struct {
		name     string
		arr      [][]int
		expected int
	}{
		{
			name:     "normal case",
			arr:      [][]int{{1, 2, 3}, {4, 5}},
			expected: 3,
		},
		{
			name:     "empty slice",
			arr:      [][]int{},
			expected: 0,
		},
		{
			name:     "all same length",
			arr:      [][]int{{1, 2}, {3, 4}, {5, 6}},
			expected: 2,
		},
		{
			name:     "single element",
			arr:      [][]int{{1, 2, 3, 4}},
			expected: 4,
		},
		{
			name:     "uneven lengths",
			arr:      [][]int{{1}, {2, 3, 4}, {5, 6}},
			expected: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.GetMaxSubSliceLength(tt.arr)
			if result != tt.expected {
				t.Errorf("GetMaxSubSliceLength() = %d, want %d", result, tt.expected)
			}
		})
	}
}

func TestSliceToInterfaceSlice(t *testing.T) {
	tests := []struct {
		name     string
		arr      []int
		validate func(t *testing.T, result []interface{})
	}{
		{
			name: "normal conversion",
			arr:  []int{1, 2, 3},
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 3 {
					t.Errorf("expected 3 elements, got %d", len(result))
				}
				if result[0].(int) != 1 || result[1].(int) != 2 || result[2].(int) != 3 {
					t.Errorf("unexpected values: %v", result)
				}
			},
		},
		{
			name: "empty slice",
			arr:  []int{},
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 0 {
					t.Errorf("expected 0 elements, got %d", len(result))
				}
			},
		},
		{
			name: "single element",
			arr:  []int{42},
			validate: func(t *testing.T, result []interface{}) {
				if len(result) != 1 {
					t.Errorf("expected 1 element, got %d", len(result))
				}
				if result[0].(int) != 42 {
					t.Errorf("unexpected value: %v", result[0])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.SliceToInterfaceSlice(tt.arr)
			tt.validate(t, result)
		})
	}
}

func TestGenerateRange(t *testing.T) {
	tests := []struct {
		name     string
		start    int
		end      int
		expected []int
	}{
		{
			name:     "normal range",
			start:    0,
			end:      5,
			expected: []int{0, 1, 2, 3, 4},
		},
		{
			name:     "start equals end",
			start:    3,
			end:      3,
			expected: []int{},
		},
		{
			name:     "positive start",
			start:    2,
			end:      7,
			expected: []int{2, 3, 4, 5, 6},
		},
		{
			name:     "single element",
			start:    0,
			end:      1,
			expected: []int{0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.GenerateRange[int](tt.start, tt.end)
			if len(result) != len(tt.expected) {
				t.Errorf("GenerateRange() = %v, want %v", result, tt.expected)
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("GenerateRange()[%d] = %d, want %d", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestStringSliceTruncatePrecisely(t *testing.T) {
	tests := []struct {
		name     string
		slices   [][]string
		maxLen   int
		expected [][]string
	}{
		{
			name:     "normal truncate",
			slices:   [][]string{{"a", "b", "c"}, {"d", "e", "f"}},
			maxLen:   4,
			expected: [][]string{{"a", "b", "c"}, {"d"}},
		},
		{
			name:     "no truncate needed",
			slices:   [][]string{{"a", "b"}, {"c", "d"}},
			maxLen:   5,
			expected: [][]string{{"a", "b"}, {"c", "d"}},
		},
		{
			name:     "truncate all",
			slices:   [][]string{{"a", "b"}, {"c", "d"}},
			maxLen:   0,
			expected: [][]string{},
		},
		{
			name:     "partial truncate",
			slices:   [][]string{{"a", "b", "c"}, {"d", "e"}, {"f"}},
			maxLen:   3,
			expected: [][]string{{"a", "b", "c"}},
		},
		{
			name:     "empty slices",
			slices:   [][]string{},
			maxLen:   5,
			expected: [][]string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.StringSliceTruncatePrecisely(tt.slices, tt.maxLen)
			if len(result) != len(tt.expected) {
				t.Errorf("StringSliceTruncatePrecisely() = %v, want %v", result, tt.expected)
			}
			for i := range result {
				if len(result[i]) != len(tt.expected[i]) {
					t.Errorf("StringSliceTruncatePrecisely()[%d] = %v, want %v", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func BenchmarkPadSlice(b *testing.B) {
	slice := make([]int, 1000)
	for i := range slice {
		slice[i] = i
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.PadSlice(slice, 1500, 0)
	}
}

func BenchmarkFlatten2DSlice(b *testing.B) {
	arr := make([][]int, 100)
	for i := range arr {
		arr[i] = make([]int, 100)
		for j := range arr[i] {
			arr[i][j] = i*100 + j
		}
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.Flatten2DSlice(arr)
	}
}

func BenchmarkSliceTransposeFor2D(b *testing.B) {
	slice := make([][]int, 50)
	for i := range slice {
		slice[i] = make([]int, 50)
		for j := range slice[i] {
			slice[i][j] = i*50 + j
		}
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.SliceTransposeFor2D(slice)
	}
}

func BenchmarkSliceTransposeFor3D(b *testing.B) {
	slice := make([][][]int, 10)
	for i := range slice {
		slice[i] = make([][]int, 10)
		for j := range slice[i] {
			slice[i][j] = make([]int, 10)
			for k := range slice[i][j] {
				slice[i][j][k] = i*100 + j*10 + k
			}
		}
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.SliceTransposeFor3D(slice)
	}
}

func BenchmarkGenerateRange(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.GenerateRange[int](0, 1000)
	}
}
