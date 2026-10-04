# vibedge

On-device bearing and imbalance fault detection from vibration — software pipeline v2.

## Quick start

```bash
make setup
make test
make figures    # synthetic smoke path, no downloads
make experiment # C1 feature-set × protocol ladder
```

## What this implements

- Physics-grounded **two-timescale** features (fast broadband + slow envelope)
- **Speed estimation** from vibration (tacho = validation only)
- **Deployment view** (6 kHz LPF, 26.7 kSPS, MEMS noise)
- Unified **6-class** taxonomy with per-dataset mappings
- Leakage ladder protocols 0–5 with **frozen hashed manifests**
- C1 hypothesis test: envelope-ratio vs time-only under increasing split rigor
- Loaders: synthetic, MaFaulDa, CWRU (KNOWN_BAD), MFPT, Paderborn

## Honest limitations

- Imbalance: held-out severity on MaFaulDa only (one rig)
- Cross-dataset transfer: bearing subset only
- MCU / energy Pareto deferred to phase 2

See [DESIGN.md](DESIGN.md).
