# Pilot 2 pre-registration (SFS-PCL v2, checkpoint 1b)

Written and committed **before any pilot-2 run**, on 2026-11-01. The principal approved it ("go").

## Why
Pilot 1 (C1-C4) and the combination test (CCVR + server momentum) failed the go rule. The CICIoT2023 fine-label
gap to SCAFFOLD points at client feature drift during local training, not at head bias. The components below act
during local training. Clients stay stateless with one model upload per round. No client control variates are used.

## Candidates

| ID | Method config | Mechanism | Own hyperparameter, tuned on seed 100 |
|---|---|---|---|
| P1 | `sfspcl_m_ntd` | SFS-PCL + server momentum (beta 0.7) + FedNTD not-true distillation (tau 1) | ntd in {0.1, 0.3, 1, 3, 10} |
| P2 | `sfspcl_cm` | SFS-PCL + FedCM: v = alpha*g + (1-alpha)*Delta_t in local steps | alpha in {0.02, 0.05, 0.1, 0.2, 0.5, 0.8} |
| P3 | `sfspcl_m_fa` | SFS-PCL + server momentum (beta 0.7) + FedFA feature anchors + classifier calibration | anchor in {0.001, 0.01, 0.1, 1, 3} |
| P4 | `sfspcl_v3x_ccvr` | X + guarded CCVR (ccvr_lr 0.1), where X is the best of P1-P3 by mean tuning-seed validation macro-F1 | none |

In P4, the guard keeps the re-fitted head only if federated validation macro-F1 improves.

## Fixed rules
- **Setting:** K=20, alpha=0.1, rho=0.5, T=50, client lr 0.05, tau 1.0. SCAFFOLD keeps lr 0.02.
- **Tuning:** seed 100, fine labels only, mean validation macro-F1 over both datasets. The grids are pre-extended; a value selected on a grid edge is reported but not extended further.
- **Evaluation seeds: 2 and 3 (fresh).** Seeds 0-1 informed the design of these components, so they are not reused for the go decision. SFS-PCL and SCAFFOLD are re-run on seeds 2-3 on the same GPU type (L4, torch 2.11.0+cu128).
- **Go rule (unchanged from the agent brief):** closure = (cand - SFS-PCL)/(SCAFFOLD - SFS-PCL) >= 50 % on fine labels on BOTH datasets, and cand - SFS-PCL >= -0.5 pp on grouped labels on both datasets (2-seed means).
- **Multiplicity:** four candidates are evaluated on two seeds, so a pass is a direction, not evidence. Any candidate that passes must still meet a success tier on the 5-seed full grid with tests.
- **Stop rule:** if no candidate passes, the beat-SCAFFOLD route ends and the work proceeds to option A (honest-framing full grid). No further component search.
- **Reporting:** every candidate tried (C1-C4, C3+C1a, P1-P4) is reported in the paper.
- **Reproducibility guard:** SFS-PCL seed 0 on CICIoT2023 fine labels is re-run after the code change and must reproduce the committed result bit-exactly.
