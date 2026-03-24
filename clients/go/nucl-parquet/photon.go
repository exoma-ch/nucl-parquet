package nuclparquet

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/parquet-go/parquet-go"
)

// xsKey is a composite map key for (Z, Process) lookups.
type xsKey struct {
	z       uint8
	process Process
}

// xsTable holds a sorted energy/cross-section table for one (element, process) pair.
type xsTable struct {
	energy []float64 // MeV, sorted ascending
	xs     []float64 // barns/atom, same length as energy
}

// tabTable holds a sorted x/y table (form factors, scattering functions).
type tabTable struct {
	x []float64
	y []float64
}

// photonXsRow is the Parquet row schema for cross-section files.
type photonXsRow struct {
	Z         int32   `parquet:"Z"`
	EnergyMeV float64 `parquet:"energy_MeV"`
	Process   string  `parquet:"process"`
	XsBarns   float64 `parquet:"xs_barns"`
}

// formFactorRow is the Parquet row schema for form factor files.
type formFactorRow struct {
	Z                int32   `parquet:"Z"`
	MomentumTransfer float64 `parquet:"momentum_transfer"`
	FormFactor       float64 `parquet:"form_factor"`
}

// scatteringFnRow is the Parquet row schema for scattering function files.
type scatteringFnRow struct {
	Z                int32   `parquet:"Z"`
	MomentumTransfer float64 `parquet:"momentum_transfer"`
	ScatteringFn     float64 `parquet:"scattering_fn"`
}

// PhotonDb is the EPDL97 photon interaction database.
//
// It is immutable after construction and safe for concurrent use from
// multiple goroutines without synchronization.
type PhotonDb struct {
	xsTables      map[xsKey]*xsTable
	formFactors   map[uint8]*tabTable
	scatteringFns map[uint8]*tabTable
}

// OpenPhotonDb loads EPDL97 data from the nucl-parquet meta/ directory.
//
// It reads:
//   - meta/epdl97/photon_xs/*.parquet — per-process cross-sections
//   - meta/epdl97/form_factors/*.parquet — Rayleigh form factors
//   - meta/epdl97/scattering_fn/*.parquet — Compton scattering functions
func OpenPhotonDb(metaDir string) (*PhotonDb, error) {
	xsDir := filepath.Join(metaDir, "epdl97", "photon_xs")
	ffDir := filepath.Join(metaDir, "epdl97", "form_factors")
	sfDir := filepath.Join(metaDir, "epdl97", "scattering_fn")

	xsTables, err := loadXsTables(xsDir)
	if err != nil {
		return nil, err
	}

	formFactors, err := loadTabTables[formFactorRow](ffDir, func(r *formFactorRow) (int32, float64, float64) {
		return r.Z, r.MomentumTransfer, r.FormFactor
	})
	if err != nil {
		return nil, err
	}

	scatteringFns, err := loadTabTables[scatteringFnRow](sfDir, func(r *scatteringFnRow) (int32, float64, float64) {
		return r.Z, r.MomentumTransfer, r.ScatteringFn
	})
	if err != nil {
		return nil, err
	}

	return &PhotonDb{
		xsTables:      xsTables,
		formFactors:   formFactors,
		scatteringFns: scatteringFns,
	}, nil
}

// CrossSection returns the interpolated cross-section [barns/atom] for element Z
// at the given energy [MeV] and interaction process.
//
// Returns 0.0 if the element or process is not in the database.
// Uses log-log interpolation on the EPDL97 energy grid.
func (db *PhotonDb) CrossSection(z uint8, energyMeV float64, process Process) float64 {
	table, ok := db.xsTables[xsKey{z, process}]
	if !ok {
		return 0.0
	}
	return LogLogInterp(table.energy, table.xs, energyMeV)
}

// AllCrossSections returns [photoelectric, compton, rayleigh, pair_total] cross-sections
// in barns/atom for element Z at the given energy [MeV].
func (db *PhotonDb) AllCrossSections(z uint8, energyMeV float64) [4]float64 {
	return [4]float64{
		db.CrossSection(z, energyMeV, Photoelectric),
		db.CrossSection(z, energyMeV, Incoherent),
		db.CrossSection(z, energyMeV, Coherent),
		db.CrossSection(z, energyMeV, PairTotal),
	}
}

// TotalCrossSection returns the total cross-section [barns/atom] for element Z
// at the given energy [MeV].
func (db *PhotonDb) TotalCrossSection(z uint8, energyMeV float64) float64 {
	return db.CrossSection(z, energyMeV, Total)
}

// FormFactor returns the Rayleigh atomic form factor for element Z at
// momentum transfer q.
//
// q = sin(theta/2) / lambda in units matching the EPDL97 table (typically 1/cm).
// Returns 0.0 if the element is not in the database.
func (db *PhotonDb) FormFactor(z uint8, q float64) float64 {
	table, ok := db.formFactors[z]
	if !ok {
		return 0.0
	}
	return LogLogInterp(table.x, table.y, q)
}

// ScatteringFunction returns the Compton incoherent scattering function S(q)
// for element Z.
//
// Ranges from 0 (forward) to Z (backward scattering).
// Returns 0.0 if the element is not in the database.
func (db *PhotonDb) ScatteringFunction(z uint8, q float64) float64 {
	table, ok := db.scatteringFns[z]
	if !ok {
		return 0.0
	}
	return LogLogInterp(table.x, table.y, q)
}

// HasElement reports whether data is loaded for element Z.
func (db *PhotonDb) HasElement(z uint8) bool {
	_, ok := db.xsTables[xsKey{z, Total}]
	return ok
}

// NumElements returns the number of elements loaded.
func (db *PhotonDb) NumElements() int {
	count := 0
	for k := range db.xsTables {
		if k.process == Total {
			count++
		}
	}
	return count
}

// loadXsTables reads all parquet files in dir and returns a map of (Z, Process) -> xsTable.
func loadXsTables(dir string) (map[xsKey]*xsTable, error) {
	tables := make(map[xsKey]*xsTable)

	info, err := os.Stat(dir)
	if err != nil {
		return nil, fmt.Errorf("data directory not found: %s", dir)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("not a directory: %s", dir)
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

		allRows, err := parquet.Read[photonXsRow](f, stat.Size())
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("parquet read error in %s: %w", path, err)
		}

		// Accumulate rows per (Z, process).
		type accumKey struct {
			z       uint8
			process Process
		}
		accum := make(map[accumKey]*xsTable)

		for i := range allRows {
			row := &allRows[i]
			proc, err := ParseProcess(row.Process)
			if err != nil {
				continue // skip unknown process
			}
			z := uint8(row.Z)
			k := accumKey{z, proc}
			tbl, ok := accum[k]
			if !ok {
				tbl = &xsTable{}
				accum[k] = tbl
			}
			tbl.energy = append(tbl.energy, row.EnergyMeV)
			tbl.xs = append(tbl.xs, row.XsBarns)
		}

		// Sort each table by energy and insert into result map.
		for k, tbl := range accum {
			sortXsTable(tbl)
			tables[xsKey{k.z, k.process}] = tbl
		}
	}

	return tables, nil
}

// loadTabTables is a generic loader for form factor / scattering function parquet files.
// The extract function pulls (Z, x, y) from each row.
func loadTabTables[R any](dir string, extract func(*R) (int32, float64, float64)) (map[uint8]*tabTable, error) {
	tables := make(map[uint8]*tabTable)

	_, err := os.Stat(dir)
	if err != nil {
		// Optional data — missing directory is not an error.
		return tables, nil
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

		allRows, err := parquet.Read[R](f, stat.Size())
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("parquet read error in %s: %w", path, err)
		}

		var zVal *uint8
		tbl := &tabTable{}

		for i := range allRows {
			z32, x, y := extract(&allRows[i])
			z := uint8(z32)
			if zVal == nil {
				zVal = &z
			}
			tbl.x = append(tbl.x, x)
			tbl.y = append(tbl.y, y)
		}

		if zVal != nil {
			sortTabTable(tbl)
			tables[*zVal] = tbl
		}
	}

	return tables, nil
}

// sortXsTable sorts an xsTable by energy ascending.
func sortXsTable(t *xsTable) {
	indices := make([]int, len(t.energy))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(a, b int) bool {
		return t.energy[indices[a]] < t.energy[indices[b]]
	})
	energy := make([]float64, len(t.energy))
	xs := make([]float64, len(t.xs))
	for i, idx := range indices {
		energy[i] = t.energy[idx]
		xs[i] = t.xs[idx]
	}
	t.energy = energy
	t.xs = xs
}

// sortTabTable sorts a tabTable by x ascending.
func sortTabTable(t *tabTable) {
	indices := make([]int, len(t.x))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(a, b int) bool {
		return t.x[indices[a]] < t.x[indices[b]]
	})
	x := make([]float64, len(t.x))
	y := make([]float64, len(t.y))
	for i, idx := range indices {
		x[i] = t.x[idx]
		y[i] = t.y[idx]
	}
	t.x = x
	t.y = y
}
