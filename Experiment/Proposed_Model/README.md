# Proposed Model: TE-Q-Transformer

This directory contains the authoritative implementation, training protocol, and evaluation pipeline for the **Temporal-Embedding Quantum Transformer (TE-Q-Transformer)** for Lithium-ion battery State of Health (SOH) estimation.

---

## 1. Architectural Summary

The TE-Q-Transformer is a hybrid quantum-classical sequence model:

1. **Input Sequence**: $[B, L=512, D=4]$ with channels:
   - $V$: Voltage (MinMax scaled on NASA train split)
   - $I$: Current (MinMax scaled on NASA train split)
   - $T_C$: Temperature in physical °C (unscaled for physical validity)
   - $t_{\text{norm}}$: Normalized cycle time $[0, 1]$
2. **Physics-Informed Quantum Embedding Layer**:
   - Arrhenius thermodynamic gate:
     $$T_K = \max(T_C + 273.15, 1.0)$$
     $$\text{SEI Term} = \exp\left(\frac{E_{a,\text{sei}} \cdot 10^4}{R}\left(\frac{1}{T_{\text{ref}}} - \frac{1}{T_K}\right)\right)$$
     $$\text{Plating Term} = \exp\left(\frac{E_{a,\text{pl}} \cdot 10^4}{R}\left(\frac{1}{T_K} - \frac{1}{T_{\text{ref}}}\right)\right)$$
     $$\phi = \text{SEI Term} + \text{Plating Term}$$
     $$\theta_T = \frac{\pi \cdot \phi}{4}$$
   - Rotations on 4 qubits ($RY$ with $[V \cdot \pi, I \cdot \pi, t \cdot \pi, \theta_T]$).
   - Rich Entangler: Layered $RZ + CNOT + IsingZZ + H + RZ$ gates.
   - Pauli-$Z$ expectation values $\langle Z_i \rangle$ read out to produce $[B \times L, 4]$.
3. **Classical Latent Projection**: Linear projection $4 \to d_{\text{model}} = 64$.
4. **Temporal Smoothing**: 1D Convolution along the sequence length (kernel size 3, padding 1).
5. **Transformer Encoder**:
   - Trainable `[CLS]` token prepended to sequence $[B, L+1, 64]$.
   - Sinusoidal 1D Positional Encoding.
   - 3-layer Transformer Encoder ($d_{\text{model}}=64$, $n_{\text{heads}}=2$, $d_{\text{ff}}=64$, GELU, dropout 0.0, norm_first).
6. **Regression Head**: `[CLS]` token pooling $\to \text{Linear}(64 \to 64) \to \text{GELU} \to \text{Linear}(64 \to 1) \to \hat{y} \in \mathbb{R}$.

---

## 2. Directory File Map

| File | Description |
| :--- | :--- |
| `model.py` | Authoritative re-export of `TEQTransformer` from `src/models/teq_transformer.py`. |
| `config.py` | Model and training hyperparameter configurations. |
| `train.py` | Training loop with AdamW, ReduceLROnPlateau, MSELoss, and early stopping. |
| `evaluate.py` | Evaluates model on NASA held-out test cells and CALCE zero-shot cells. |
| `run.py` | Command-line interface for running evaluation or training. |

---

## 3. How to Run

### Evaluate Pre-trained Checkpoint (Default)
To evaluate the frozen NASA baseline without re-training:

```bash
# Evaluate on both NASA and CALCE
python Experiment/Proposed_Model/run.py --dataset all

# Evaluate on NASA only
python Experiment/Proposed_Model/run.py --dataset NASA

# Evaluate on CALCE zero-shot only
python Experiment/Proposed_Model/run.py --dataset CALCE
```

### Train Model from Scratch
```bash
python Experiment/Proposed_Model/run.py --train --epochs 80 --batch-size 8
```

Outputs are automatically saved into `Experiment/Result/NASA/` and `Experiment/Result/CALCE/`.
