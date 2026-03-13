/** Photon interaction process types (EPDL97). */
export enum Process {
  /** Total cross-section (sum of all processes) */
  Total = "total",
  /** Coherent (Rayleigh) scattering */
  Coherent = "coherent",
  /** Incoherent (Compton) scattering */
  Incoherent = "incoherent",
  /** Photoelectric absorption (total) */
  Photoelectric = "photoelectric",
  /** Pair production — nuclear field */
  PairNuclear = "pair_nuclear",
  /** Pair production — electron field */
  PairElectron = "pair_electron",
  /** Pair production — total (nuclear + electron) */
  PairTotal = "pair_total",
}

/** Type of atomic relaxation transition. */
export enum TransitionType {
  /** Radiative transition — emits a characteristic X-ray photon. */
  Radiative = "radiative",
  /** Auger transition — emits an electron. */
  Auger = "auger",
}

/** A single atomic relaxation transition (EADL). */
export interface Transition {
  /** Shell where the vacancy was created (e.g., "K", "L1"). */
  vacancyShell: string;
  /** Shell that fills the vacancy (e.g., "L3"). */
  fillingShell: string;
  /** Radiative (X-ray) or Auger (electron). */
  transitionType: TransitionType;
  /** Transition energy in keV. */
  energyKeV: number;
  /** Transition probability (fractional, sums to ~1 per shell). */
  probability: number;
  /** Binding energy of the vacancy shell in keV. */
  edgeKeV: number;
}
