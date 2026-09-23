//go:build go1.27 && goexperiment.simd && (amd64 || arm64)

package transformers

import (
	"bytes"
	"encoding/binary"
	"math"
	"simd/archsimd"
	"strconv"
	"testing"
)

func TestPutInt32LittleEndianSIMD(test *testing.T) {
	lanes := (archsimd.Int32x4{}).Len()
	for _, length := range []int{0, 1, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 128, 129} {
		test.Run(strconv.Itoa(length), func(test *testing.T) {
			source := make([]int32, length)
			for index := range source {
				source[index] = int32(index*0x1010101) ^ -1
			}
			if length >= 2 {
				source[0] = math.MinInt32
				source[1] = math.MaxInt32
			}

			encoded := make([]byte, length*4)
			putInt32LittleEndian(encoded, source)

			expected := make([]byte, length*4)
			for index, value := range source {
				binary.LittleEndian.PutUint32(expected[index*4:], uint32(value))
			}
			if !bytes.Equal(encoded, expected) {
				test.Fatalf("putInt32LittleEndian() = %x, want %x", encoded, expected)
			}
		})
	}
}
