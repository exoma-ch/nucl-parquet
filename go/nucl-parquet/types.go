package nuclparquet

import "fmt"

// Process represents a photon interaction process type.
type Process int

const (
	// Total cross-section (sum of all processes).
	Total Process = iota
	// Coherent (Rayleigh) scattering.
	Coherent
	// Incoherent (Compton) scattering.
	Incoherent
	// Photoelectric absorption (total).
	Photoelectric
	// PairNuclear is pair production in the nuclear field.
	PairNuclear
	// PairElectron is pair production in the electron field.
	PairElectron
	// PairTotal is total pair production (nuclear + electron).
	PairTotal
)

// String returns the lowercase string representation of a Process.
func (p Process) String() string {
	switch p {
	case Total:
		return "total"
	case Coherent:
		return "coherent"
	case Incoherent:
		return "incoherent"
	case Photoelectric:
		return "photoelectric"
	case PairNuclear:
		return "pair_nuclear"
	case PairElectron:
		return "pair_electron"
	case PairTotal:
		return "pair_total"
	default:
		return fmt.Sprintf("Process(%d)", int(p))
	}
}

// ParseProcess converts a string to a Process.
// Returns an error if the string is not a recognized process name.
func ParseProcess(s string) (Process, error) {
	switch s {
	case "total":
		return Total, nil
	case "coherent":
		return Coherent, nil
	case "incoherent":
		return Incoherent, nil
	case "photoelectric":
		return Photoelectric, nil
	case "pair_nuclear":
		return PairNuclear, nil
	case "pair_electron":
		return PairElectron, nil
	case "pair_total":
		return PairTotal, nil
	default:
		return 0, fmt.Errorf("unknown process: %q", s)
	}
}

// TransitionType distinguishes radiative (X-ray) from non-radiative (Auger) transitions.
type TransitionType int

const (
	// Radiative transition — emits a characteristic X-ray photon.
	Radiative TransitionType = iota
	// Auger transition — emits an electron.
	Auger
)

// String returns the lowercase string representation of a TransitionType.
func (t TransitionType) String() string {
	switch t {
	case Radiative:
		return "radiative"
	case Auger:
		return "auger"
	default:
		return fmt.Sprintf("TransitionType(%d)", int(t))
	}
}

// Transition represents a single atomic relaxation transition.
type Transition struct {
	// VacancyShell is the shell where the vacancy was created (e.g., "K", "L1").
	VacancyShell string
	// FillingShell is the shell that fills the vacancy (e.g., "L3").
	FillingShell string
	// Type is Radiative (X-ray) or Auger (electron).
	Type TransitionType
	// EnergyKeV is the transition energy in keV.
	EnergyKeV float64
	// Probability is the transition probability (fractional, sums to ~1 per shell).
	Probability float64
	// EdgeKeV is the binding energy of the vacancy shell in keV.
	EdgeKeV float64
}
