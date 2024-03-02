package utils

// PadSlice [1, 2, 3, 4], padLength = 6, paddingValue = 0 ==> [1, 2, 3, 4, 0, 0]
func PadSlice[T any](slice []T, padLength int, paddingValue T) []T {
	if len(slice) >= padLength {
		// If the slice length is already greater than or equal to the padding length,
		// no padding is required.
		return slice
	}

	// Creating filled slices
	paddedSlice := make([]T, padLength)
	copy(paddedSlice, slice)

	// fill value
	for i := len(slice); i < padLength; i++ {
		paddedSlice[i] = paddingValue
	}

	return paddedSlice
}

// StringSliceTruncate truncate uses heuristic of trimming seq with longest len until sequenceLen satisfied.
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

// SliceTransposeFor3D Transport 3-D Dimension Slice. Like NxM to MxN.
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

// SliceTransposeFor2D Transport 2-D Dimension Slice. Like NxM to MxN.
func SliceTransposeFor2D[T any](slice [][]T) [][]T {
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

// Flatten2DSlice [][]T to []T
func Flatten2DSlice[T any](arr [][]T) []T {
	var result []T

	for i := range arr {
		result = append(result, arr[i]...)
	}

	return result
}

// GetMaxSubSliceLength [][]int{{1,2,3}, {3, 4}} return 2
func GetMaxSubSliceLength[T any](arr [][]T) (maxLength int) {
	if len(arr) == 0 {
		return 0
	}

	for _, row := range arr {
		if len(row) > maxLength {
			maxLength = len(row)
		}
	}

	return maxLength
}

// SliceToInterfaceSlice any slice to []interface{}.
func SliceToInterfaceSlice[T any](arr []T) []interface{} {
	result := make([]interface{}, len(arr))
	for i := range arr {
		result[i] = arr[i]
	}
	return result
}
