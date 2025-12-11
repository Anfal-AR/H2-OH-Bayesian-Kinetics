# Data Dictionary

## H₂ + OH → H₂O + H Kinetics Analysis

---

## Overview

| Property | Value |
|----------|-------|
| **Reaction** | H₂ + OH → H₂O + H |
| **Studies Analyzed** | 10 independent investigations |
| **Date Range** | 1981–2021 (40 years) |
| **Temperature Coverage** | 200–3044 K |
| **Data Source** | NIST Chemical Kinetics Database |

---

## Data Files

### 1. `arrhenius_parameters.csv`

Contains modified Arrhenius parameters from each study.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `Study` | String | — | Author and year (e.g., "Yang2021") |
| `Year` | Integer | — | Publication year |
| `T_min` | Float | K | Minimum valid temperature |
| `T_max` | Float | K | Maximum valid temperature |
| `A` | Float | cm³ molecule⁻¹ s⁻¹ | Pre-exponential factor |
| `n` | Float | — | Temperature exponent (dimensionless) |
| `Ea` | Float | kJ/mol | Activation energy |
| `Method` | String | — | Experimental technique |
| `Uncertainty` | Float | % | Reported uncertainty |
| `Priority` | String | — | High/Medium/Low (based on quality) |

**Sample Data:**

| Study | Year | T_min | T_max | A | n | Ea | Method | Uncertainty |
|-------|------|-------|-------|---|---|-----|--------|-------------|
| Yang2021 | 2021 | 300 | 2500 | 1.54e-12 | 1.64 | 13.72 | Review/Statistical | 15% |
| Varga2016 | 2016 | 800 | 2500 | 9.79e-13 | 1.88 | 13.19 | Shock tube | 20% |
| Hong2010 | 2010 | 1014 | 3044 | 8.79e-13 | 2.08 | 14.72 | Shock tube | 25% |
| Atkinson2004 | 2004 | 200 | 450 | 7.70e-12 | 0.00 | 17.46 | Review | 15% |
| Baulch1992 | 1992 | 300 | 2500 | 1.55e-12 | 1.60 | 13.80 | Review | 20% |

---

### 2. `kinetics_database.csv`

Calculated rate constants at standard temperatures for all studies.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `Temperature` | Float | K | Temperature point |
| `Study` | String | — | Study identifier |
| `k` | Float | cm³ molecule⁻¹ s⁻¹ | Rate constant |
| `log10_k` | Float | — | Log₁₀ of rate constant |
| `In_Range` | Boolean | — | Whether T is within study's valid range |

**Temperature Points:** 250, 300, 400, 500, 600, 750, 800, 1000, 1200, 1500, 2000, 2500, 3000 K

---

### 3. `bayesian_posteriors.csv`

Results of Bayesian uncertainty quantification.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `Temperature` | Float | K | Temperature point |
| `N_Studies` | Integer | — | Number of studies contributing |
| `Posterior_Mean` | Float | cm³ molecule⁻¹ s⁻¹ | Bayesian posterior mean |
| `Posterior_Std` | Float | cm³ molecule⁻¹ s⁻¹ | Posterior standard deviation |
| `Empirical_Std` | Float | cm³ molecule⁻¹ s⁻¹ | Empirical standard deviation |
| `Total_Std` | Float | cm³ molecule⁻¹ s⁻¹ | Total uncertainty |
| `CI_Lower_95` | Float | cm³ molecule⁻¹ s⁻¹ | 95% credible interval lower bound |
| `CI_Upper_95` | Float | cm³ molecule⁻¹ s⁻¹ | 95% credible interval upper bound |
| `Relative_Uncertainty` | Float | % | (Total_Std / Posterior_Mean) × 100 |
| `Measurement_Contribution` | Float | % | σ²_measurement / σ²_total × 100 |
| `InterStudy_Contribution` | Float | % | σ²_inter-study / σ²_total × 100 |

---

### 4. `sensitivity_analysis.csv`

Parameter sensitivity coefficients at each temperature.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `Temperature` | Float | K | Temperature point |
| `S_A` | Float | — | Sensitivity to pre-exponential factor |
| `S_n` | Float | — | Sensitivity to temperature exponent |
| `S_Ea` | Float | — | Sensitivity to activation energy |
| `Abs_S_A` | Float | — | Absolute value of S_A |
| `Abs_S_n` | Float | — | Absolute value of S_n |
| `Abs_S_Ea` | Float | — | Absolute value of S_Ea |
| `Dominant_Parameter` | String | — | Parameter with highest |S| |
| `Uncertainty_from_A` | Float | % | Contribution to total uncertainty |
| `Uncertainty_from_n` | Float | % | Contribution to total uncertainty |
| `Uncertainty_from_Ea` | Float | % | Contribution to total uncertainty |
| `Total_Propagated_Uncertainty` | Float | % | Combined uncertainty |

**Sensitivity Formulas:**

For modified Arrhenius: k(T) = A × (T/298)ⁿ × exp(-Ea/RT)

```
S_A = ∂ln(k)/∂ln(A) = 1 (constant)
S_n = ∂ln(k)/∂ln(n) = n × ln(T/298)
S_Ea = ∂ln(k)/∂ln(Ea) = -Ea/(RT)
```

---

### 5. `ml_model_results.csv`

Machine learning model performance metrics.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `Model` | String | — | Model name |
| `R2_Train` | Float | — | Training R² score |
| `R2_Test` | Float | — | Test R² score |
| `RMSE_Train` | Float | — | Training RMSE |
| `RMSE_Test` | Float | — | Test RMSE |
| `MAE_Test` | Float | — | Test mean absolute error |
| `MAPE_Test` | Float | % | Test mean absolute percentage error |
| `N_Parameters` | Integer | — | Number of model parameters |

---

## Variable Definitions

### Rate Constant (k)

The rate constant for a bimolecular reaction:

```
Rate = k × [H₂] × [OH]

Units: cm³ molecule⁻¹ s⁻¹

Conversion: 1 cm³ molecule⁻¹ s⁻¹ = 6.022 × 10²³ cm³ mol⁻¹ s⁻¹
                                  = 6.022 × 10²⁰ L mol⁻¹ s⁻¹
```

### Modified Arrhenius Parameters

| Parameter | Symbol | Units | Typical Range | Description |
|-----------|--------|-------|---------------|-------------|
| Pre-exponential factor | A | cm³ molecule⁻¹ s⁻¹ | 10⁻¹³ to 10⁻¹¹ | Collision frequency factor |
| Temperature exponent | n | dimensionless | 0 to 2.1 | Power-law temperature dependence |
| Activation energy | Ea | kJ/mol | 13–18 | Energy barrier for reaction |

### Physical Constants

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Gas constant | R | 8.314472 × 10⁻³ | kJ mol⁻¹ K⁻¹ |
| Reference temperature | T₀ | 298 | K |

---

## Study Details

### Experimental Methods

| Method | Studies | Description |
|--------|---------|-------------|
| **Shock Tube** | Old1992, Hong2010, Varga2016 | High-T measurements using shock-heated gases |
| **Flash Photolysis** | Ravishankara1981, Demissy1997 | UV photolysis to generate radicals |
| **Laser-Induced Fluorescence** | Pirraglia1989 | LIF detection of OH radicals |
| **Laser Photolysis** | Sutherland1996 | Laser-based radical generation |
| **Review/Evaluation** | Baulch1992, Atkinson2004, Yang2021 | Critical compilation of multiple sources |

### Priority Classification

| Priority | Criteria | Studies |
|----------|----------|---------|
| **High** | Recent, wide T-range, comprehensive | Yang2021, Varga2016, Hong2010, Baulch1992, Old1992 |
| **Medium** | Older but reliable, limited T-range | Atkinson2004, Sutherland1996, Pirraglia1989 |
| **Low** | Very limited T-range, older methods | Ravishankara1981, Demissy1997 |

---

## Temperature Zones

| Zone | T Range (K) | Application | N Studies | Avg Uncertainty |
|------|-------------|-------------|-----------|-----------------|
| **Atmospheric** | 200–450 | Troposphere/stratosphere chemistry | 7 | 20.7% |
| **Intermediate** | 450–800 | Transition regime | 5 | 12.3% |
| **Combustion** | 800–2000 | Engines, burners, reactors | 4–5 | **5.8%** |
| **High-T** | 2000–3044 | Shock tubes, detonations | 2–3 | 14.3% |

---

## Usage Notes

### Loading Data in Python

```python
import pandas as pd

# Load Arrhenius parameters
params = pd.read_csv('data/arrhenius_parameters.csv')

# Load Bayesian posteriors
posteriors = pd.read_csv('data/bayesian_posteriors.csv')

# Load sensitivity analysis
sensitivity = pd.read_csv('data/sensitivity_analysis.csv')
```

### Calculating Rate Constant

```python
import numpy as np

def modified_arrhenius(T, A, n, Ea, R=8.314472e-3):
    """
    Calculate rate constant using modified Arrhenius expression.
    
    Parameters:
    -----------
    T : float or array
        Temperature (K)
    A : float
        Pre-exponential factor (cm³ molecule⁻¹ s⁻¹)
    n : float
        Temperature exponent (dimensionless)
    Ea : float
        Activation energy (kJ/mol)
    R : float
        Gas constant (kJ mol⁻¹ K⁻¹)
    
    Returns:
    --------
    k : float or array
        Rate constant (cm³ molecule⁻¹ s⁻¹)
    """
    return A * (T / 298.0)**n * np.exp(-Ea / (R * T))

# Example: Yang et al. (2021) at 1000 K
k = modified_arrhenius(1000, A=1.54e-12, n=1.64, Ea=13.72)
print(f"k(1000 K) = {k:.2e} cm³ molecule⁻¹ s⁻¹")
```

---

## Quality Control

### Data Validation

- ✅ All parameters verified against original publications
- ✅ Cross-checked with NIST database values
- ✅ Unit conversions validated
- ✅ Temperature range boundaries confirmed

### Known Issues

- Demissy1997: Very limited temperature range (200–300 K)
- Hong2010: Extrapolation above 3000 K not validated
- Standard Arrhenius (n=0): Only valid for narrow T-ranges

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01 | Initial release |

---

## Contact

For questions about this dataset:
- **Author:** Anfal Rababah
- **Email:** Anfal0Rababah@gmail.com
- **ORCID:** 0009-0003-7450-8907
