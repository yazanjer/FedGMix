
## Pilot 3 notes

- Pre-registered in docs/PILOT3_PREREG.md (commit 4cec104), with an implementation-only addendum written before any result
  (GPU EM, mixture cache, vectorised sampler; commit 20d6af5). Fresh evaluation seeds 5-6.
- Reproducibility guard passed again (SFS-PCL seed 0, CICIoT2023 fine: 0.5660143665331144, bit-exact).
- Tuning (seed 100, mean validation macro-F1): r = 1.0 selected (0.702). All ratios scored 0.697-0.702, so the result is
  insensitive to r. Reference points: SFS-PCL 0.637, SFS-PCL + momentum 0.679, CCVR 0.716.
- The guard kept the re-fitted CCVR head in 6 of 8 runs (it rejected it on CICIoT2023 fine seed 5 and Edge-IIoTset grouped seed 5).
- Upload per run, fine labels: CICIoT2023 SCAFFOLD 225.2 MB, SFS-PCL 112.6, V 115.9, V+C 121.7.
  Edge-IIoTset: 270.4, 135.2, 142.4, 145.0. V+C uses about 54 % of SCAFFOLD's upload.
- **Result: V + guarded CCVR PASSES the go rule** (closure 82 % / 145 %; grouped +0.22 / +5.92 pp vs SFS-PCL).
  V alone fails only on CICIoT2023 grouped labels (-1.25 pp).
- Caveats: two seeds only. SCAFFOLD varies a lot between these seeds (CICIoT2023 fine 63.2 / 55.0; Edge-IIoTset fine
  74.3 / 87.2). This is the 11th candidate across pilots 1-3. A pass is a direction; the 5-seed full grid with tests decides.
- Pod fsp2-pilot3 (L4, $0.59/h): 2.02 h, about $1.19. SFS-PCL v2 total: about $3.7.
