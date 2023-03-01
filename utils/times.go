package utils

import "time"

// GetNanoTimeFromSys get nano timestamp
func GetNanoTimeFromSys() int64 {
	return time.Now().UnixNano()
}

// CalTimeGapWithNS get nano timestamp gap
func CalTimeGapWithNS(begin int64) int64 {
	return GetNanoTimeFromSys() - begin
}
