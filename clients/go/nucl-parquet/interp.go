package nuclparquet

import (
	"math"
	"sort"
)

// LogLogInterp performs log-log interpolation on sorted x/y arrays.
//
// Both slices must be the same length and sorted by x (ascending).
// Values are clamped to the first/last table entry when outside range.
// Returns NaN if the slice is empty or energy <= 0.
func LogLogInterp(xs, ys []float64, x float64) float64 {
	n := len(xs)
	if n == 0 || x <= 0 {
		return math.NaN()
	}

	// Clamp to table range.
	if x <= xs[0] {
		return ys[0]
	}
	if x >= xs[n-1] {
		return ys[n-1]
	}

	// Binary search for the bracketing interval.
	idx := sort.SearchFloat64s(xs, x)

	// Exact match.
	if idx < n && xs[idx] == x {
		return ys[idx]
	}

	// x is between xs[idx-1] and xs[idx].
	i := idx - 1
	x0, x1 := xs[i], xs[i+1]
	y0, y1 := ys[i], ys[i+1]

	// If either value is zero or negative, fall back to linear interpolation.
	if y0 <= 0 || y1 <= 0 {
		t := (x - x0) / (x1 - x0)
		return y0 + t*(y1-y0)
	}

	// Log-log interpolation.
	logX := math.Log(x)
	logX0 := math.Log(x0)
	logX1 := math.Log(x1)
	logY0 := math.Log(y0)
	logY1 := math.Log(y1)

	t := (logX - logX0) / (logX1 - logX0)
	return math.Exp(logY0 + t*(logY1-logY0))
}
