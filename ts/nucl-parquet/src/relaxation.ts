/**
 * EADL atomic relaxation data (fluorescence X-rays and Auger electrons).
 *
 * After a photoelectric interaction creates a vacancy in an inner shell,
 * the atom relaxes via radiative (X-ray) or non-radiative (Auger) transitions.
 * This module provides the transition probabilities and energies needed to
 * sample the resulting secondary particles.
 */

import { parquetRead } from "hyparquet";
import { Transition, TransitionType } from "./types.js";

/** Load file as ArrayBuffer — Node.js implementation. */
async function loadFileAsArrayBuffer(path: string): Promise<ArrayBuffer> {
  const { readFile } = await import("fs/promises");
  const buffer = await readFile(path);
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

/** Read a Parquet file and return rows. */
async function readParquetFile(path: string): Promise<Record<string, unknown>[][]> {
  const arrayBuffer = await loadFileAsArrayBuffer(path);
  const rows: Record<string, unknown>[][] = [];
  await parquetRead({
    file: {
      byteLength: arrayBuffer.byteLength,
      slice(start: number, end: number) {
        return arrayBuffer.slice(start, end);
      },
    },
    onComplete(data: Record<string, unknown>[][]) {
      rows.push(...data);
    },
  });
  return rows;
}

/** List .parquet files in a directory. */
async function listParquetFiles(dir: string): Promise<string[]> {
  const { readdir } = await import("fs/promises");
  const { join } = await import("path");
  const entries = await readdir(dir);
  return entries
    .filter((e) => e.endsWith(".parquet"))
    .map((e) => join(dir, e));
}

/** Check if a directory exists. */
async function dirExists(dir: string): Promise<boolean> {
  try {
    const { stat } = await import("fs/promises");
    const s = await stat(dir);
    return s.isDirectory();
  } catch {
    return false;
  }
}

/**
 * EADL atomic relaxation database.
 *
 * Use the static `open()` factory method to create an instance.
 *
 * @example
 * ```ts
 * const db = await RelaxationDb.open("path/to/nucl-parquet/meta");
 * const fy = db.fluorescenceYield(29, "K"); // ~0.44
 * ```
 */
export class RelaxationDb {
  private transitionMap: Map<number, Transition[]>;

  private constructor(transitionMap: Map<number, Transition[]>) {
    this.transitionMap = transitionMap;
  }

  /**
   * Load EADL data from the nucl-parquet `meta/` directory.
   *
   * Reads `meta/eadl/*.parquet`.
   */
  static async open(metaDir: string): Promise<RelaxationDb> {
    const { join } = await import("path");
    const dir = join(metaDir, "eadl");
    const transitionMap = new Map<number, Transition[]>();

    if (!(await dirExists(dir))) {
      return new RelaxationDb(transitionMap);
    }

    const files = await listParquetFiles(dir);
    for (const file of files) {
      const rows = await readParquetFile(file);

      let zVal: number | undefined;
      const transList: Transition[] = [];

      for (const rowGroup of rows) {
        for (const row of rowGroup) {
          const z = row["Z"] as number;
          if (zVal === undefined) zVal = z;

          transList.push({
            vacancyShell: row["vacancy_shell"] as string,
            fillingShell: row["filling_shell"] as string,
            transitionType:
              (row["transition_type"] as string) === "radiative"
                ? TransitionType.Radiative
                : TransitionType.Auger,
            energyKeV: row["energy_keV"] as number,
            probability: row["probability"] as number,
            edgeKeV: row["edge_keV"] as number,
          });
        }
      }

      if (zVal !== undefined) {
        // Sort by shell, then probability descending
        transList.sort((a, b) => {
          const shellCmp = a.vacancyShell.localeCompare(b.vacancyShell);
          if (shellCmp !== 0) return shellCmp;
          return b.probability - a.probability;
        });
        transitionMap.set(zVal, transList);
      }
    }

    return new RelaxationDb(transitionMap);
  }

  /** Get all transitions for element Z. */
  transitions(z: number): Transition[] {
    return this.transitionMap.get(z) ?? [];
  }

  /** Get transitions for a specific vacancy shell (e.g., "K"). */
  shellTransitions(z: number, shell: string): Transition[] {
    return this.transitions(z).filter((t) => t.vacancyShell === shell);
  }

  /** Get only radiative (X-ray) transitions for element Z and shell. */
  radiativeTransitions(z: number, shell: string): Transition[] {
    return this.transitions(z).filter(
      (t) => t.vacancyShell === shell && t.transitionType === TransitionType.Radiative,
    );
  }

  /** Fluorescence yield for a shell = sum of radiative transition probabilities. */
  fluorescenceYield(z: number, shell: string): number {
    return this.transitions(z)
      .filter((t) => t.vacancyShell === shell && t.transitionType === TransitionType.Radiative)
      .reduce((sum, t) => sum + t.probability, 0);
  }

  /** Check if data is loaded for element Z. */
  hasElement(z: number): boolean {
    return this.transitionMap.has(z);
  }
}
