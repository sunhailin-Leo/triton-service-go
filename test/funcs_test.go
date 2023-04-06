package test

import (
	"testing"

	"github.com/sunhailin-Leo/triton-service-go/utils"
)

func TestIsWhiteSpace(t *testing.T) {
	if !utils.IsWhitespace(' ') {
		t.Errorf("function IsWhitespace get error result")
	}
	if utils.IsWhitespace('a') {
		t.Errorf("function IsWhitespace get error result")
	}
}

func TestIsControl(t *testing.T) {
	if utils.IsControl('\t') {
		t.Errorf("function IsControl get error result")
	}
}
