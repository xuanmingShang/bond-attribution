"""
Bond PnL Attribution System
============================
A fixed-income analytics system for U.S. Treasury bonds.

Modules
-------
- yield_curve : FRED data loading, interpolation, curve management
- bond        : Bond specification, pricing, cashflow scheduling
- attribution : Campisi-style PnL decomposition
- pca         : PCA analysis and PCA-based attribution
- ladder      : Bond ladder backtest
- main        : CLI entry-point and demo runner
"""
