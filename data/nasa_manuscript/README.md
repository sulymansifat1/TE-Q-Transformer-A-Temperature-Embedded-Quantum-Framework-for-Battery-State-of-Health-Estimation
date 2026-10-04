# Stored NASA held-out results

These files are the cycle-level records behind the manuscript tables. The notebooks recompute metrics from them. They are not new experiments.

- `nasa_heldout_predictions.csv`: seed-42 predictions for TE-Q-Transformer and the ten baselines on B0018 (132), B0032 (39), and the B0053 test portion (16). TE-Q-Transformer predictions match `Experiment/Result/NASA/predictions/E01_TEQ_*` within float32 rounding.
- `explainability/per_sample_attributions.csv`: Integrated Gradients on the same 187 cycles, 50-step Gauss–Legendre quadrature, training-mean voltage/current baseline, temperature baseline 25 °C, and time baseline 0.
- `explainability/channel_ablation_results.csv`: frozen-model input replacement.
- `explainability/permutation_importance_results.csv`: ten cycle-wise shuffles.
- `explainability/temperature_vs_soh_bins.csv`: attribution bins saved with that analysis.
