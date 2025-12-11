# Bayesian Uncertainty Quantification for H₂ + OH → H₂O + H Reaction Kinetics

[![DOI](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2025--rd9v8-blue.svg)](https://doi.org/10.26434/chemrxiv-2025-rd9v8)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Open Science](https://img.shields.io/badge/Open-Science-green.svg)](https://en.wikipedia.org/wiki/Open_science)

## 📄 Overview

This repository contains the complete analysis code, datasets, and supplementary materials for the paper **"Bayesian Uncertainty Quantification and Sensitivity Analysis for the H₂ + OH → H₂O + H Reaction: A Comprehensive Comparison of Ten Kinetic Studies"** (Rababah, 2025).

The H₂ + OH → H₂O + H reaction is fundamental to **hydrogen combustion**, **atmospheric chemistry**, and **clean energy systems**. This study provides the first comprehensive Bayesian uncertainty quantification and sensitivity analysis for this critical elementary reaction.

### 🎯 Key Contributions

- **Bayesian uncertainty quantification** with hierarchical decomposition of measurement vs. inter-study variability
- **Sensitivity analysis** identifying which Arrhenius parameters dominate at different temperatures
- **Machine learning validation** confirming modified Arrhenius form captures true physics
- **Application-specific recommendations** for atmospheric chemistry, combustion, and high-T applications
- **Complete open-source workflow** for reproducible chemical kinetics analysis

---

## 📊 Study Summary

| Aspect | Details |
|--------|---------|
| **Studies Analyzed** | 10 independent investigations (1981–2021) |
| **Temperature Range** | 200–3044 K (span: 2844 K) |
| **Methods** | Bayesian inference, sensitivity analysis, ML validation |
| **Data Source** | NIST Chemical Kinetics Database |
| **Average Uncertainty** | 14.6% (range: 10.0%–21.2%) |
| **Best Agreement Zone** | Combustion (800–2000 K): CV < 6% |

---

## 🔬 Key Findings

### Bayesian Posterior Estimates

| Temperature | Posterior Mean k | 95% CI | Uncertainty | N Studies |
|-------------|------------------|--------|-------------|-----------|
| 300 K | 6.85 × 10⁻¹⁵ | ± 1.42 × 10⁻¹⁵ | 20.7% | 7 |
| 500 K | 1.33 × 10⁻¹³ | ± 1.35 × 10⁻¹⁴ | 10.0% | 4 |
| 750 K | 6.61 × 10⁻¹³ | ± 1.40 × 10⁻¹³ | 21.2% | 4 |
| 1000 K | 2.09 × 10⁻¹² | ± 2.33 × 10⁻¹³ | **5.8%** | 5 |
| 1500 K | 7.21 × 10⁻¹² | ± 7.27 × 10⁻¹³ | 10.1% | 4 |
| 2000 K | 1.56 × 10⁻¹¹ | ± 2.12 × 10⁻¹² | 13.7% | 3 |
| 2500 K | 2.66 × 10⁻¹¹ | ± 4.86 × 10⁻¹² | 18.3% | 2 |

*Units: cm³ molecule⁻¹ s⁻¹*

### Uncertainty Decomposition

| Temperature | Measurement | Inter-study | Total | Dominant Source |
|-------------|-------------|-------------|-------|-----------------|
| 300 K | 3.2% | 17.5% | 20.7% | Inter-study |
| 500 K | 8.9% | 1.1% | 10.0% | Measurement |
| 1000 K | 2.1% | 3.7% | 5.8% | Balanced |
| 2000 K | 5.1% | 8.6% | 13.7% | Inter-study |

**Key Insight:** Inter-study variability dominates at most temperatures, indicating systematic differences between experimental methods.

### Sensitivity Analysis Summary

| Temperature | Dominant Parameter | \|S_Ea\| | \|S_n\| | \|S_A\| |
|-------------|-------------------|----------|---------|---------|
| 300 K | **Activation Energy (Ea)** | 6.8 | 0.0 | 1.0 |
| 700 K | Transition | 3.2 | 1.7 | 1.0 |
| 1500 K | **Temperature Exponent (n)** | 2.1 | 2.7 | 1.0 |
| 2500 K | **Temperature Exponent (n)** | 1.3 | 3.4 | 1.0 |

**Practical Implications:**
- **Atmospheric chemistry (T < 500 K):** Prioritize accurate Ea measurements
- **Combustion (800–2000 K):** All three parameters matter; balanced accuracy needed
- **High-T applications (T > 2000 K):** Temperature exponent n is critical

### Machine Learning Validation

| Model | R² (Test) | MAPE (%) | Parameters |
|-------|-----------|----------|------------|
| Polynomial Ridge | 0.9993 | 6.0 | 56 |
| Random Forest | 0.9985 | 8.6 | 100+ |
| Gradient Boosting | 0.9983 | 8.9 | 100+ |
| Neural Network | 0.9917 | 22.2 | 2,500+ |
| **Modified Arrhenius** | **0.9981** | **10.3** | **3** |

**Conclusion:** ML models provide minimal improvement over 3-parameter Arrhenius (ΔR² < 0.002), confirming physical appropriateness of the traditional form.

---

## 📈 Visualizations

### Figure 1: Comprehensive Study Comparison
![Study Comparison](results/figures/Figure1_study_comparison.png)

*Ten kinetic studies spanning 200–3044 K showing excellent agreement at combustion temperatures and greater scatter at extremes.*

### Figure 2: Bayesian Uncertainty Quantification
![Bayesian Analysis](results/figures/Figure2_bayesian_analysis.png)

*Posterior estimates with 95% credible intervals showing U-shaped uncertainty pattern with minimum at 1000 K.*

### Figure 3: Parameter Sensitivity Analysis
![Sensitivity Analysis](results/figures/Figure3_sensitivity_analysis.png)

*Activation energy dominates at low T; temperature exponent becomes critical above 1500 K.*

### Figure 4: Temperature Zone Recommendations
![Zone Analysis](results/figures/Figure4_zone_recommendations.png)

*Application-specific recommendations with data coverage and uncertainty by temperature zone.*

### Figure 5: Machine Learning Validation
![ML Validation](results/figures/Figure5_ml_validation.png)

*ML models confirm modified Arrhenius captures true physics—additional complexity provides minimal benefit.*

---

## 🚀 Quick Start

### Option 1: Run Complete Analysis

```bash
# Clone the repository
git clone https://github.com/YourUsername/H2-OH-Bayesian-Kinetics.git
cd H2-OH-Bayesian-Kinetics

# Install dependencies
pip install -r code/requirements.txt

# Run main analysis (generates all figures and data)
python code/H2_OH_comprehensive_analysis.py
```

### Option 2: Interactive Exploration

```python
import numpy as np

# Modified Arrhenius function
def modified_arrhenius(T, A, n, Ea, R=8.314472e-3):
    """Calculate rate constant k(T) = A × (T/298)^n × exp(-Ea/RT)"""
    return A * (T / 298.0)**n * np.exp(-Ea / (R * T))

# Yang et al. (2021) - Recommended for combustion modeling
k_1000K = modified_arrhenius(1000, A=1.54e-12, n=1.64, Ea=13.72)
print(f"k(1000 K) = {k_1000K:.2e} cm³ molecule⁻¹ s⁻¹")
# Output: k(1000 K) = 2.16e-12 cm³ molecule⁻¹ s⁻¹
```

### Option 3: Adapt for Your Reaction

1. Replace `data/arrhenius_parameters.csv` with your reaction's data
2. Update temperature ranges in configuration
3. Run the same analysis pipeline
4. Get Bayesian uncertainties for your system!

---

## 📐 Methodology

### Bayesian Framework

**Hierarchical Uncertainty Decomposition:**
```
σ²_total = σ²_measurement + σ²_inter-study

where:
• σ²_measurement = posterior variance from inverse-variance weighting
• σ²_inter-study = empirical variance between studies
```

### Sensitivity Analysis

Normalized sensitivity coefficients for modified Arrhenius k(T) = A × (T/298)ⁿ × exp(-Ea/RT):
```
S_A  = 1              (constant)
S_n  = ln(T/298)      (increases with T)
S_Ea = -Ea/(RT)       (decreases with T)
```

### Machine Learning Validation

**Purpose:** Confirm Arrhenius form captures physics

**Logic:** If data followed a different functional form, ML models would show **much** better R² than Arrhenius. Since the improvement is minimal (<0.2%), the Arrhenius form is physically appropriate.

---

## 📖 Rate Expressions

### Recommended Correlations by Application

| Application | Temperature | Recommended Study | Rate Expression |
|-------------|-------------|-------------------|-----------------|
| **Atmospheric Chemistry** | 200–450 K | Atkinson et al. (2004) | k = 7.70×10⁻¹² exp(-2100/T) |
| **Combustion Modeling** | 800–2000 K | Yang et al. (2021) | k = 1.54×10⁻¹² (T/298)^1.64 exp(-1651/T) |
| **High-Temperature** | 2000–3044 K | Hong et al. (2010) | k = 8.79×10⁻¹³ (T/298)^2.08 exp(-1771/T) |

*Note: Exponential term shown as exp(-Ea/R/T) where Ea/R values are: 2100 K (17.46 kJ/mol), 1651 K (13.72 kJ/mol), 1771 K (14.72 kJ/mol)*

---

## 🔧 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

---

## 📚 Citation

If you use this analysis, code, or data in your research, please cite:

```bibtex
@article{rababah2025bayesian,
  title={Bayesian Uncertainty Quantification and Sensitivity Analysis for the 
         H₂ + OH → H₂O + H Reaction: A Comprehensive Comparison of Ten Kinetic Studies},
  author={Rababah, Anfal},
  journal={ChemRxiv},
  year={2025},
  doi={10.26434/chemrxiv-2025-rd9v8},
  url={https://doi.org/10.26434/chemrxiv-2025-rd9v8}
}
```

**APA Format:**
> Rababah, A. (2025). Bayesian uncertainty quantification and sensitivity analysis for the H₂ + OH → H₂O + H reaction: A comprehensive comparison of ten kinetic studies. *ChemRxiv*. https://doi.org/10.26434/chemrxiv-2025-rd9v8

---

## 🔗 Data Sources

- **NIST Chemical Kinetics Database:** https://kinetics.nist.gov/
- **Original Publications:** See Table 1 in paper for complete references (10 studies, 1981–2021)

---

## 📜 License

This work is licensed under the **Creative Commons Attribution 4.0 International License** (CC BY 4.0).

You are free to share and adapt this material for any purpose with appropriate attribution.

---

## 👤 Author

**Anfal Rababah**
- 📧 Email: Anfal0Rababah@gmail.com
- 🔬 ORCID: [0009-0003-7450-8907](https://orcid.org/0009-0003-7450-8907)

---

## 🙏 Acknowledgments

- **Chemical Kinetics Community** for four decades of meticulous experimental work (1981–2021)
- **NIST** for maintaining the Chemical Kinetics Database
- **Research Groups:** Ravishankara, Pirraglia, Baulch, Old, Sutherland, Demissy & Lesclaux, Atkinson, Hong, Varga, and Yang
- **Claude (Anthropic)** for assistance with code development and manuscript preparation

---

<p align="center">
  <strong>Bayesian Methods • Chemical Kinetics • Open Science</strong>
</p>

<p align="center">
  Made with ❤️ for the combustion and atmospheric chemistry communities
</p>
