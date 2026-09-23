//go:build go1.27 && goexperiment.simd && gc && (amd64 || arm64)

package transformers

import (
	"encoding/binary"
	"simd/archsimd"
)

func putInt32LittleEndian(destination []byte, source []int32) {
	lanes := (archsimd.Int32x4{}).Len()
	index := 0
	for ; index+lanes <= len(source); index += lanes {
		archsimd.LoadInt32x4(source[index:]).ToBits().ReshapeToUint8s().Store(destination[index*4:])
	}
	for ; index < len(source); index++ {
		binary.LittleEndian.PutUint32(destination[index*4:], uint32(source[index]))
	}
}
