package nuclparquet

import (
	"math"
	"testing"
)

func TestLogLogInterpExactMatch(t *testing.T) {
	e := []float64{1.0, 10.0, 100.0}
	v := []float64{100.0, 10.0, 1.0}
	got := LogLogInterp(e, v, 10.0)
	if got != 10.0 {
		t.Errorf("exact match: got %v, want 10.0", got)
	}
}

func TestLogLogInterpPowerLaw(t *testing.T) {
	// For a perfect power law v = 1000/E, log-log interp is exact.
	e := []float64{1.0, 1000.0}
	v := []float64{1000.0, 1.0}
	got := LogLogInterp(e, v, 10.0)
	if math.Abs(got-100.0) > 1e-10 {
		t.Errorf("power law: got %v, want 100.0", got)
	}
}

func TestLogLogInterpClampBelow(t *testing.T) {
	e := []float64{1.0, 10.0}
	v := []float64{100.0, 10.0}
	got := LogLogInterp(e, v, 0.5)
	if got != 100.0 {
		t.Errorf("clamp below: got %v, want 100.0", got)
	}
}

func TestLogLogInterpClampAbove(t *testing.T) {
	e := []float64{1.0, 10.0}
	v := []float64{100.0, 10.0}
	got := LogLogInterp(e, v, 20.0)
	if got != 10.0 {
		t.Errorf("clamp above: got %v, want 10.0", got)
	}
}

func TestLogLogInterpEmpty(t *testing.T) {
	got := LogLogInterp(nil, nil, 1.0)
	if !math.IsNaN(got) {
		t.Errorf("empty: got %v, want NaN", got)
	}
}

func TestLogLogInterpZeroEnergy(t *testing.T) {
	e := []float64{1.0, 10.0}
	v := []float64{100.0, 10.0}
	got := LogLogInterp(e, v, 0.0)
	if !math.IsNaN(got) {
		t.Errorf("zero energy: got %v, want NaN", got)
	}
}

func TestLogLogInterpNegativeEnergy(t *testing.T) {
	e := []float64{1.0, 10.0}
	v := []float64{100.0, 10.0}
	got := LogLogInterp(e, v, -1.0)
	if !math.IsNaN(got) {
		t.Errorf("negative energy: got %v, want NaN", got)
	}
}

func TestLogLogInterpLinearFallback(t *testing.T) {
	// When a value is zero, should fall back to linear interpolation.
	e := []float64{1.0, 2.0}
	v := []float64{0.0, 10.0}
	got := LogLogInterp(e, v, 1.5)
	want := 5.0
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("linear fallback: got %v, want %v", got, want)
	}
}
