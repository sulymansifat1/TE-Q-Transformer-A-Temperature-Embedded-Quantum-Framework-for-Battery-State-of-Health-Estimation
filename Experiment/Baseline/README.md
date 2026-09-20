# Baseline Model Benchmark Suite (Experiment E05)

This directory implements the locked baseline benchmark suite for Lithium-ion battery SOH estimation under identical data contracts and evaluation protocols.

---

## 1. Locked Baseline Set (NASA SOH Estimation)

| Model | File | Architecture Overview | Scientific Identity & Role |
| :--- | :--- | :--- | :--- |
| **LSTM** | `LSTM.py` | 2-layer standard LSTM ($d_{\text{model}}=64$), last-token pooling, MLP head. | Canonical recurrent baseline (Hochreiter & Schmidhuber, 1997). |
| **GRU** | `GRU.py` | 2-layer Gated Recurrent Unit ($d_{\text{model}}=64$), last-token pooling, MLP head. | Canonical lightweight recurrent baseline (Cho et al., 2014). |
| **CNN1D** | `CNN1D.py` | 3-layer 1D Temporal CNN with BatchNorm, GELU, and global average pooling. | Pure convolutional feature extractor. |
| **TCN** | `TCN.py` | Dilated causal 1D convolutions with residual blocks and exponential dilations ($2^0 \dots 2^3$). | Long-range causal sequence model (Bai et al., 2018; `locuslab/TCN`). |
| **DLinear** | `DLinear.py` | Moving-average series decomposition into trend and seasonal components + linear heads. | Canonical LTSF decomposition linear baseline (Zeng et al., AAAI 2023; `cure-lab/LTSF-Linear`). |
| **Transformer**| `Transformer.py` | 3-layer classical Transformer Encoder ($d=64, h=2$), `[CLS]` token pooling. | Pure multi-head self-attention encoder baseline (Vaswani et al., 2017). |
| **PatchTST** | `PatchTST.py` | Sub-series patching ($PL=16, S=8$) + channel-independent Transformer encoder + RevIN. | SOTA patching time-series model (Nie et al., ICLR 2023; `yuqinie98/PatchTST`). |
| **iTransformer**| `iTransformer.py` | Inverted tokenization (variates as tokens) + cross-variate Transformer encoder. | SOTA inverted transformer model (Liu et al., ICLR 2024; `thuml/iTransformer`). |
| **QLSTM** | `QLSTM.py` | Gate-level Variational Quantum Circuits embedded in all 4 LSTM gates ($i, f, g, o$). | Quantum-classical recurrent baseline (Chen et al., ICASSP 2022; `ycchen1989/QLSTM`). |
| **QGRU** | `QGRU.py` | Gate-level Variational Quantum Circuits embedded in GRU gates with shared FC boundaries and circular CNOTs. | Quantum-classical recurrent baseline (Ceschini et al., J. Phys. Commun. 2024). |

---

## 2. Safety & Verification Protocol

Per strict publication audit policy:
- Full benchmark training is not launched without completed source-integrity audits and verified dry-runs.
- Dry-run validation is performed via `Experiment/Ablation_Study/E05_baseline_comparison.py --mode dry_run`.
- All models operate under the identical NASA contract:
  - Input: $[B, 512, 4]$ (`Voltage_V`, `Current_A`, `Temperature_C`, `Time_norm`)
  - Target: scalar SOH $[B]$
  - Split: Train (B0005, B0006, B0007, B0029, B0030, B0031, first 70% B0053); Test (B0018, B0032, final 30% B0053).
  - Scaler: Train-only MinMax normalization on Voltage, Current, Time_norm; unscaled Celsius on Temperature.
