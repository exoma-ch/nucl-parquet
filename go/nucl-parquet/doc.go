// Package nuclparquet provides fast, thread-safe access to nuclear interaction
// data stored as Parquet files.
//
// Designed as the physics data backbone for Monte Carlo particle transport codes.
// All data structures are immutable after construction and safe for concurrent use.
//
// # Data sources
//
//   - EPDL97: Photon cross-sections (photoelectric, Compton, Rayleigh, pair production)
//   - EADL: Atomic relaxation (fluorescence X-ray and Auger transition probabilities)
//
// # Usage
//
//	db, err := nuclparquet.OpenPhotonDb("path/to/nucl-parquet/meta")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	xs := db.CrossSection(29, 0.511, nuclparquet.Photoelectric) // barns/atom
//	ff := db.FormFactor(29, 0.5) // dimensionless
package nuclparquet
