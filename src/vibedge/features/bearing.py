"""Bearing characteristic frequencies and band energy integration.

Kinematic idealisations — real bearings slip 1–2%, so integrate energy over
±tol bands rather than peak-picking a single FFT bin.

Sanity identities (asserted in tests):
  BPFO + BPFI = n * f_r
  BPFO = n * FTF

Note: a ball defect appears at 2×BSF, not BSF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BearingGeometry:
    n: int  # rolling elements
    d: float  # ball diameter
    D: float  # pitch diameter
    phi_deg: float = 0.0  # contact angle (degrees)

    @property
    def phi(self) -> float:
        return np.deg2rad(self.phi_deg)

    @property
    def rd(self) -> float:
        return (self.d / self.D) * np.cos(self.phi)


# Common geometries used in public datasets (approximate catalogue values)
CWRU_6205_DE = BearingGeometry(n=9, d=0.3126, D=1.537, phi_deg=0.0)  # inches
CWRU_6203_FE = BearingGeometry(n=9, d=0.2656, D=1.122, phi_deg=0.0)
# MFPT test rig (MFPT data-set README): 8 balls, 0.235" ball, 1.245" pitch, 0°
MFPT_BEARING = BearingGeometry(n=8, d=0.235, D=1.245, phi_deg=0.0)  # inches
# Paderborn KAt: 6203 deep-groove (Lessmeier et al. 2016)
PADERBORN_6203 = BearingGeometry(n=8, d=6.75, D=28.55, phi_deg=0.0)  # mm
# MaFaulDa MFS rig (UFRJ page): 8 balls, 0.7145 cm ball, 2.8519 cm cage/pitch
# → BPFO 2.998, BPFI 5.002 × fr, matching the published table.
MAFAULDA_BEARING = BearingGeometry(n=8, d=0.7145, D=2.8519, phi_deg=0.0)  # cm
# Generic deep-groove used for synthetic / MaFaulDa when exact geo unknown
GENERIC_6205 = BearingGeometry(n=9, d=7.94, D=39.04, phi_deg=0.0)  # mm


@dataclass(frozen=True)
class FaultFrequencies:
    fr: float
    bpfo: float
    bpfi: float
    ftf: float
    bsf: float
    ball_fault: float  # 2 * BSF

    def as_dict(self) -> dict[str, float]:
        return {
            "fr": self.fr,
            "BPFO": self.bpfo,
            "BPFI": self.bpfi,
            "FTF": self.ftf,
            "BSF": self.bsf,
            "2xBSF": self.ball_fault,
        }


def bearing_freqs(fr: float, geo: BearingGeometry) -> FaultFrequencies:
    """Compute BPFO, BPFI, FTF, BSF, 2×BSF for shaft rate fr (Hz)."""
    if fr <= 0:
        raise ValueError(f"fr must be positive, got {fr}")
    rd = geo.rd
    n = geo.n
    bpfo = (n * fr / 2.0) * (1.0 - rd)
    bpfi = (n * fr / 2.0) * (1.0 + rd)
    ftf = (fr / 2.0) * (1.0 - rd)
    bsf = (geo.D * fr / (2.0 * geo.d)) * (1.0 - rd**2)
    return FaultFrequencies(
        fr=fr, bpfo=bpfo, bpfi=bpfi, ftf=ftf, bsf=bsf, ball_fault=2.0 * bsf
    )


def assert_identities(ff: FaultFrequencies, geo: BearingGeometry, atol: float = 1e-9) -> None:
    n_fr = geo.n * ff.fr
    if not np.isclose(ff.bpfo + ff.bpfi, n_fr, atol=atol):
        raise AssertionError(
            f"BPFO+BPFI={ff.bpfo + ff.bpfi} != n*fr={n_fr}"
        )
    if not np.isclose(ff.bpfo, geo.n * ff.ftf, atol=atol):
        raise AssertionError(f"BPFO={ff.bpfo} != n*FTF={geo.n * ff.ftf}")


def band_energy(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    center_hz: float,
    tol: float = 0.02,
) -> float:
    """Integrate |spectrum| energy in ±tol relative band around center_hz."""
    if center_hz <= 0:
        return 0.0
    lo = center_hz * (1.0 - tol)
    hi = center_hz * (1.0 + tol)
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        # fall back to nearest bin
        idx = int(np.argmin(np.abs(freqs - center_hz)))
        return float(spectrum[idx] ** 2)
    return float(np.sum(spectrum[mask] ** 2))


def harmonic_band_energies(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    fundamental: float,
    n_harmonics: int = 3,
    tol: float = 0.02,
) -> list[float]:
    return [
        band_energy(freqs, spectrum, fundamental * h, tol=tol)
        for h in range(1, n_harmonics + 1)
    ]


def sideband_pair_energy(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    carrier: float,
    spacing: float,
    tol: float = 0.02,
) -> float:
    """Energy at carrier±spacing (pair sum)."""
    return band_energy(freqs, spectrum, carrier - spacing, tol) + band_energy(
        freqs, spectrum, carrier + spacing, tol
    )
