package test

import (
	"testing"
	"time"

	"github.com/sunhailin-Leo/triton-service-go/v2/utils"
)

func TestGetNanoTimeFromSys(t *testing.T) {
	nanoTime := utils.GetNanoTimeFromSys()
	if nanoTime <= 0 {
		t.Errorf("expected positive nano time, got %d", nanoTime)
	}

	// Should be close to current time
	now := time.Now().UnixNano()
	diff := now - nanoTime
	if diff < 0 {
		diff = -diff
	}
	// Should be within 1 second
	if diff > int64(time.Second) {
		t.Errorf("nano time too far from current time, diff: %d ns", diff)
	}
}

func TestCalTimeGapWithNS(t *testing.T) {
	begin := utils.GetNanoTimeFromSys()
	time.Sleep(1 * time.Millisecond)
	gap := utils.CalTimeGapWithNS(begin)

	if gap <= 0 {
		t.Errorf("expected positive time gap, got %d", gap)
	}
	// Should be at least 1ms
	if gap < int64(time.Millisecond) {
		t.Errorf("expected gap >= 1ms, got %d ns", gap)
	}
}
