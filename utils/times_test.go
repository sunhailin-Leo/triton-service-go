package utils_test

import (
	"testing"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

func TestGetNanoTimeFromSys(t *testing.T) {
	result := utils.GetNanoTimeFromSys()

	if result <= 0 {
		t.Errorf("GetNanoTimeFromSys() returned non-positive value: %d", result)
	}

	now := time.Now().UnixNano()
	diff := now - result

	if diff < 0 {
		diff = -diff
	}

	if diff > 1000000000 {
		t.Errorf("GetNanoTimeFromSys() returned value too far from current time: diff = %d ns", diff)
	}
}

func TestCalTimeGapWithNS(t *testing.T) {
	begin := utils.GetNanoTimeFromSys()
	time.Sleep(10 * time.Millisecond)

	result := utils.CalTimeGapWithNS(begin)

	if result <= 0 {
		t.Errorf("CalTimeGapWithNS() returned non-positive value: %d", result)
	}

	if result < 10000000 {
		t.Errorf("CalTimeGapWithNS() returned value too small: %d ns (expected at least 10ms)", result)
	}

	if result > 200000000 {
		t.Errorf("CalTimeGapWithNS() returned value too large: %d ns (expected ~10ms)", result)
	}
}

func BenchmarkGetNanoTimeFromSys(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.GetNanoTimeFromSys()
	}
}

func BenchmarkCalTimeGapWithNS(b *testing.B) {
	begin := utils.GetNanoTimeFromSys()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		utils.CalTimeGapWithNS(begin)
	}
}
