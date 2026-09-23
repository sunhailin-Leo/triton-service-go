//go:build !go1.27 || !goexperiment.simd || (!amd64 && !arm64)

package transformers

import "encoding/binary"

func putInt32LittleEndian(destination []byte, source []int32) {
	for index, value := range source {
		binary.LittleEndian.PutUint32(destination[index*4:], uint32(value))
	}
}
