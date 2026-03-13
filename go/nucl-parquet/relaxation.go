package nuclparquet

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/parquet-go/parquet-go"
)

// relaxationRow is the Parquet row schema for EADL relaxation files.
type relaxationRow struct {
	Z              int32   `parquet:"Z"`
	VacancyShell   string  `parquet:"vacancy_shell"`
	FillingShell   string  `parquet:"filling_shell"`
	TransitionType string  `parquet:"transition_type"`
	EnergyKeV      float64 `parquet:"energy_keV"`
	Probability    float64 `parquet:"probability"`
	EdgeKeV        float64 `parquet:"edge_keV"`
}

// RelaxationDb is the EADL atomic relaxation database.
//
// It is immutable after construction and safe for concurrent use from
// multiple goroutines without synchronization.
type RelaxationDb struct {
	transitions map[uint8][]Transition
}

// OpenRelaxationDb loads EADL data from the nucl-parquet meta/ directory.
//
// Reads meta/eadl/*.parquet.
func OpenRelaxationDb(metaDir string) (*RelaxationDb, error) {
	dir := filepath.Join(metaDir, "eadl")
	transitions := make(map[uint8][]Transition)

	_, err := os.Stat(dir)
	if err != nil {
		// Missing directory is not an error — just no data.
		return &RelaxationDb{transitions: transitions}, nil
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".parquet" {
			continue
		}
		path := filepath.Join(dir, entry.Name())

		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}

		stat, err := f.Stat()
		if err != nil {
			f.Close()
			return nil, err
		}

		pf, err := parquet.OpenFile(f, stat.Size())
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("parquet read error in %s: %w", path, err)
		}

		var zVal *uint8
		var transList []Transition

		for _, rg := range pf.RowGroups() {
			rows := rg.NumRows()
			buf := make([]relaxationRow, rows)
			n, err := rg.Rows().Read(buf)
			if err != nil {
				f.Close()
				return nil, fmt.Errorf("read error in %s: %w", path, err)
			}
			for i := 0; i < n; i++ {
				row := &buf[i]
				z := uint8(row.Z)
				if zVal == nil {
					zVal = &z
				}

				tt := Auger
				if row.TransitionType == "radiative" {
					tt = Radiative
				}

				transList = append(transList, Transition{
					VacancyShell: row.VacancyShell,
					FillingShell: row.FillingShell,
					Type:         tt,
					EnergyKeV:    row.EnergyKeV,
					Probability:  row.Probability,
					EdgeKeV:      row.EdgeKeV,
				})
			}
		}

		f.Close()

		if zVal != nil {
			// Sort by vacancy shell, then probability descending.
			sort.Slice(transList, func(i, j int) bool {
				if transList[i].VacancyShell != transList[j].VacancyShell {
					return transList[i].VacancyShell < transList[j].VacancyShell
				}
				return transList[i].Probability > transList[j].Probability
			})
			transitions[*zVal] = transList
		}
	}

	return &RelaxationDb{transitions: transitions}, nil
}

// Transitions returns all transitions for element Z.
// Returns nil if the element is not in the database.
func (db *RelaxationDb) Transitions(z uint8) []Transition {
	return db.transitions[z]
}

// ShellTransitions returns transitions for a specific vacancy shell (e.g., "K").
func (db *RelaxationDb) ShellTransitions(z uint8, shell string) []Transition {
	all := db.transitions[z]
	var result []Transition
	for i := range all {
		if all[i].VacancyShell == shell {
			result = append(result, all[i])
		}
	}
	return result
}

// RadiativeTransitions returns only radiative (X-ray) transitions for element Z and shell.
func (db *RelaxationDb) RadiativeTransitions(z uint8, shell string) []Transition {
	all := db.transitions[z]
	var result []Transition
	for i := range all {
		if all[i].VacancyShell == shell && all[i].Type == Radiative {
			result = append(result, all[i])
		}
	}
	return result
}

// FluorescenceYield returns the fluorescence yield for a shell, which is the
// sum of radiative transition probabilities.
func (db *RelaxationDb) FluorescenceYield(z uint8, shell string) float64 {
	all := db.transitions[z]
	var sum float64
	for i := range all {
		if all[i].VacancyShell == shell && all[i].Type == Radiative {
			sum += all[i].Probability
		}
	}
	return sum
}

// HasElement reports whether data is loaded for element Z.
func (db *RelaxationDb) HasElement(z uint8) bool {
	_, ok := db.transitions[z]
	return ok
}
