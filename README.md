# CausalGraphMemoryNet (CGMN): Neural Time Series Forecasting with Learnable Causal Graphs

CGMN is a novel time series forecasting architecture that explicitly learns causal relationships between variables while maintaining temporal memory for pattern retrieval. Unlike traditional approaches that treat multivariate channels independently or use fixed graph structures, CGMN dynamically discovers variable dependencies through differentiable graph learning and integrates them with temporal attention mechanisms.

## Overview
Most time series models either ignore inter-variable relationships or assume they're static. CGMN introduces a paradigm shift: it **jointly learns causal graphs and temporal patterns** in a unified framework. The model discovers which variables influence each other, stores reusable temporal patterns in memory banks, and shares learned representations across layers—all while maintaining parameter efficiency.

This approach is particularly powerful for datasets where variables exhibit dynamic dependencies (e.g., sensor networks, electricity grids, traffic systems) that evolve with temporal context.

## Key Innovations

- **Graph-Gated Channel Fusion (GGCF)**: Learns directed causal relationships between variables through differentiable adjacency matrices; gates information flow based on learned dependencies rather than hand-crafted features

- **Temporal Memory Bank (TMB)**: Maintains a learnable set of temporal pattern prototypes that can be retrieved via attention; enables few-shot adaptation to recurring patterns without retraining

- **Progressive Weight Sharing (PWS)**: Dynamically selects from a pool of weight templates across layers; reduces redundancy while preserving model expressiveness

- **End-to-End Differentiability**: All components (graph structure, memory retrieval, weight selection) are jointly optimized with the forecasting objective

## Code Structure
the notebook contains:
1. **Configuration**: Hyperparameters for architecture and training
2. **Graph Module** (`GraphGatedChannelFusion`): Learnable causal graph with gating
3. **Memory Module** (`TemporalMemoryBank`): Multi-head attention over learnable memory slots
4. **Sharing Module** (`SharedWeightFFN`): Template-based weight generation
5. **Encoder Stack**: Modular layers combining all three innovations
6. **Training Pipeline**: Multi-seed evaluation with ablation studies
7. **Analysis Tools**: Graph sparsity, memory utilization, and sharing statistics

## Requirements
- Python 3.8+, PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- CUDA-capable GPU (tested on Tesla P100)

## Experimental Results

### Forecasting Performance
- **Consistent Improvement**: Outperforms TSMixer, PatchTST, Transformer, and traditional RNN baselines
- **Low Variance**: Stable performance across multiple random seeds
- **Competitive Parameters**: Achieves strong results with ~50k parameters

### Ablation Studies
All three components contribute measurably:
- **w/o GGCF**: Performance degrades without causal graph learning
- **w/o TMB**: Removing temporal memory hurts pattern recognition
- **w/o PWS**: Disabling weight sharing increases redundancy

### Model Interpretability
1. **Learned Causal Structure**: Graph sparsity reveals which variable pairs have strong dependencies
2. **Memory Utilization**: Tracks which temporal patterns are frequently accessed
3. **Weight Sharing Efficiency**: Quantifies parameter reuse across layers

## Novel Contributions
- an architecture to **jointly optimize causal graphs and temporal memory** for time series
- Demonstrates that **learnable graphs outperform fixed or no-graph baselines**
- Introduces **memory-augmented attention with causal masking** for temporal modeling
- Validates **progressive weight sharing** as an effective regularization technique
