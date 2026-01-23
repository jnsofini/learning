# LGD Modeling Methodology Whitepaper
## Comparing Binary Transformation vs. Continuous Target Approaches

---

**Prepared for:** Credit Risk Team  
**Date:** January 2026  
**Classification:** Internal Use

---

## Executive Summary

**Purpose:** This whitepaper evaluates two methodologies for Loss Given Default (LGD) scorecard development and provides a recommendation for future model development initiatives.

**Key Finding:** Modern continuous target approaches can deliver comparable discriminatory power to traditional binary transformation methods while eliminating 100× data duplication and significantly reducing computational complexity.

**Recommendation:** [PLACEHOLDER: To be completed based on results]

**Estimated Benefits:**
- Development time reduction: [PLACEHOLDER: X%]
- Memory/storage savings: [PLACEHOLDER: X GB/TB]
- Simplified governance and validation processes
- Maintained or improved model performance

---

## Table of Contents

1. [Background and Business Context](#1-background-and-business-context)
2. [Current State: Binary Transformation Approach](#2-current-state-binary-transformation-approach)
3. [Proposed Alternative: Continuous Target Approach](#3-proposed-alternative-continuous-target-approach)
4. [Comparative Analysis](#4-comparative-analysis)
5. [Empirical Results](#5-empirical-results)
6. [Risk Assessment](#6-risk-assessment)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Governance and Validation](#8-governance-and-validation)
9. [Recommendations](#9-recommendations)
10. [Appendices](#10-appendices)

---

## 1. Background and Business Context

### 1.1 The LGD Modeling Challenge

Loss Given Default (LGD) represents the percentage of exposure lost when a borrower defaults. Under Basel regulatory frameworks, accurate LGD estimation is critical for:

- **Capital Adequacy:** Determining risk-weighted assets (RWA)
- **Economic Capital:** Internal capital allocation
- **Expected Loss Provisioning:** IFRS 9 / CECL compliance
- **Risk-Based Pricing:** Setting appropriate loan terms
- **Portfolio Management:** Strategic decision-making

**The Core Challenge:** Unlike Probability of Default (PD), which is binary (default/non-default), LGD is continuous (0-100% loss severity), making it incompatible with traditional binary classification scorecards.

### 1.2 Current Practice Industry-Wide

Many institutions have adopted workarounds to leverage binary scorecard infrastructure:

1. **Binary Transformation:** Converting continuous LGD to pseudo-binary through data duplication
2. **Discretization:** Bucketing LGD into high/low categories (loses granularity)
3. **Direct Regression:** Using standard regression without scorecard framework (loses WoE benefits)

Our current methodology employs Binary Transformation, which this whitepaper evaluates against modern alternatives.

### 1.3 Business Drivers for Review

- **Computational Constraints:** Increasing portfolio sizes strain current 100× expansion approach
- **Model Development Cycles:** Pressure to reduce time-to-market for model updates
- **Regulatory Evolution:** Growing acceptance of advanced continuous modeling techniques
- **Technology Advancement:** New tools (OptBinning) enable continuous WoE encoding
- **Operational Efficiency:** Need to optimize resource utilization across the model development lifecycle

---

## 2. Current State: Binary Transformation Approach

### 2.1 Methodology Overview

**Process Flow:**

```
Original Dataset (N observations)
         ↓
Data Expansion (N × 100 observations)
         ↓
WoE Binning on Expanded Binary Target
         ↓
Logistic Regression Scorecard
         ↓
Score All Expanded Records
         ↓
Collapse to Original Dataset (Average Scores)
         ↓
Segmentation & Validation
```

### 2.2 Detailed Technical Process

**Step 1: Data Expansion**
- Each account with LGD value duplicated 100 times
- If LGD = 0.35 → 35 records with target=1, 65 records with target=0
- Expansion factor: 100× (e.g., 50,000 accounts → 5,000,000 records)

**Step 2: Feature Engineering**
- Apply Weight of Evidence (WoE) binning to each predictor
- Uses expanded binary dataset (5M records)
- Calculate Information Value (IV) for feature selection
- Enforce monotonicity constraints where appropriate

**Step 3: Model Development**
- Logistic regression on binary target using WoE-transformed features
- Coefficient estimation on expanded dataset
- Convert to scorecard format with scaled points

**Step 4: Scoring & Aggregation**
- Score all 100 duplicated records per original account
- Average scores for each original account
- Final score represents predicted LGD

**Step 5: Segmentation**
- Segment accounts into risk bands based on average scores
- Validate segment LGD distributions
- Monitor segment stability over time

### 2.3 Current Approach Strengths

| Strength | Description | Business Impact |
|----------|-------------|-----------------|
| **Familiar Metrics** | AUC, KS, Gini widely understood | Easy stakeholder communication |
| **Established Infrastructure** | Leverages existing binary scorecard tools | Low implementation risk |
| **Regulatory Precedent** | Similar to PD approaches regulators understand | Smoother approval process |
| **Team Expertise** | Current team highly proficient with methodology | Minimal training required |
| **Proven Track Record** | Successfully deployed in production | Low execution risk |

### 2.4 Current Approach Pain Points

| Pain Point | Impact | Severity |
|------------|--------|----------|
| **Data Expansion** | 100× storage requirements | High |
| **Processing Time** | [PLACEHOLDER: X hours] for model fitting | Medium-High |
| **Memory Constraints** | Cannot process full portfolio in-memory | High |
| **Development Cycles** | Extended iteration time during model tuning | Medium |
| **Validation Complexity** | Difficult to explain artificial sample size inflation | Medium |
| **Infrastructure Costs** | Higher compute and storage costs | Medium |

### 2.5 Current Performance Baseline

**Model Performance (Latest Validation):**
- AUC: [PLACEHOLDER: X.XX]
- KS Statistic: [PLACEHOLDER: X.XX]
- Gini Coefficient: [PLACEHOLDER: X.XX]
- MSE: [PLACEHOLDER: X.XX]
- MAE: [PLACEHOLDER: X.XX]

**Operational Metrics:**
- Model development time: [PLACEHOLDER: X weeks]
- Model fitting time: [PLACEHOLDER: X hours]
- Peak memory usage: [PLACEHOLDER: X GB]
- Production scoring time: [PLACEHOLDER: X minutes for Y accounts]

---

## 3. Proposed Alternative: Continuous Target Approach

### 3.1 Methodology Overview

**Process Flow:**

```
Original Dataset (N observations)
         ↓
Continuous WoE Binning (No Expansion)
         ↓
Regression Scorecard on Continuous Target
         ↓
Score Original Records Directly
         ↓
Segmentation & Validation
```

### 3.2 Detailed Technical Process

**Step 1: Feature Engineering with Continuous WoE**
- Apply ContinuousOptimalBinning to each predictor
- Calculate surrogate WoE based on deviation from mean LGD
- No data duplication—work with original 50,000 accounts

**Continuous WoE Formula:**
```
WoE_bin_i = |mean(LGD)_bin_i - mean(LGD)_population|
```

**Step 2: Model Development**
- Regression model (linear, Huber, Tobit, or Beta) on continuous LGD target
- Uses WoE-transformed features
- Coefficient estimation on original dataset

**Step 3: Scorecard Creation**
- Convert regression coefficients to scorecard points
- Apply scaling (e.g., min-max to 0-1000 range)
- Generate scorecard identical in format to binary approach

**Step 4: Scoring**
- Score original observations directly (no averaging needed)
- Produces LGD predictions on continuous scale

**Step 5: Segmentation**
- Segment based on score ranges
- Validate segment characteristics
- Compare to binary approach segments

### 3.3 Technical Foundation: OptBinning for Continuous Targets

**Key Capability:** OptBinning library (version 0.19+) extends WoE encoding to continuous targets.

**How It Works:**
1. Optimal binning identifies splits that maximize target separation
2. For each bin, calculates mean LGD deviation from population mean
3. Maintains monotonicity and business logic constraints
4. Produces WoE values suitable for regression modeling

**Code Example:**
```python
from optbinning import ContinuousOptimalBinning

binning = ContinuousOptimalBinning(
    name='debt_to_income',
    dtype='numerical',
    monotonic_trend='ascending'
)
binning.fit(X['debt_to_income'], y_lgd)
X_woe = binning.transform(X['debt_to_income'], metric='mean')
```

### 3.4 Proposed Approach Advantages

| Advantage | Description | Business Impact |
|-----------|-------------|-----------------|
| **No Data Duplication** | Work with original dataset size | Immediate 100× storage reduction |
| **Faster Processing** | Reduced computational burden | [PLACEHOLDER: X% faster] |
| **Simpler Pipeline** | Fewer transformation steps | Lower operational risk |
| **Direct Modeling** | True continuous target distribution | Better theoretical foundation |
| **Scalability** | Can handle larger portfolios | Future-proofs infrastructure |
| **Easier Governance** | True sample sizes in documentation | Clearer audit trail |

### 3.5 Proposed Approach Considerations

| Consideration | Mitigation Strategy |
|---------------|---------------------|
| **Unfamiliar Metrics** | Provide training on continuous WoE interpretation; continue reporting binary metrics for comparison |
| **Less Regulatory Precedent** | Document alignment with advanced IRB approaches; prepare detailed methodology documentation |
| **Team Training Needed** | Phased implementation with knowledge transfer; maintain dual capabilities initially |
| **Validation Framework Update** | Develop new validation templates; leverage both continuous and binary metrics during transition |

---

## 4. Comparative Analysis

### 4.1 Analysis Framework

**Objective:** Empirically compare the two approaches across multiple dimensions to determine if the continuous approach can deliver equivalent business value with reduced complexity.

**Evaluation Criteria:**

1. **Discriminatory Power:** Does the model separate high vs. low LGD effectively?
2. **Segmentation Consistency:** Do both approaches produce similar risk segments?
3. **Predictive Accuracy:** Which approach better predicts actual realized LGD?
4. **Operational Efficiency:** Time, memory, and resource requirements
5. **Stability:** Performance consistency across time periods
6. **Explainability:** Ease of interpretation and governance

### 4.2 Test Design

**Dataset:**
- Training Period: [PLACEHOLDER: Date range]
- Validation Period: [PLACEHOLDER: Date range]
- Out-of-Time Test: [PLACEHOLDER: Date range]
- Sample Size: [PLACEHOLDER: N defaults]

**Modeling Approach:**
- **Binary Transformation:** Expand data 100×, develop logistic scorecard
- **Continuous Target:** Use OptBinning continuous WoE, develop regression scorecard

**Same Predictors:** Both models use identical feature sets for fair comparison

**Segmentation Strategy:** Both models segmented into 5 risk bands:
- Very Low Risk (0-20th percentile)
- Low Risk (20-40th percentile)
- Medium Risk (40-60th percentile)
- High Risk (60-80th percentile)
- Very High Risk (80-100th percentile)

### 4.3 Metrics Comparison Framework

| Dimension | Binary Approach Metrics | Continuous Approach Metrics |
|-----------|------------------------|---------------------------|
| **Discrimination** | AUC, KS, Gini | Rank correlation (Kendall's τ, Spearman's ρ), CLAR |
| **Accuracy** | Brier Score | MSE, MAE, RMSE |
| **Calibration** | Hosmer-Lemeshow | Residual plots, calibration slope |
| **Segmentation** | Mean LGD by segment, KS between segments | Mean LGD by segment, ANOVA F-statistic |
| **Stability** | PSI (Population Stability Index) | PSI, CSI (Characteristic Stability Index) |
| **Efficiency** | Processing time, memory usage | Processing time, memory usage |

---

## 5. Empirical Results

### 5.1 Data Efficiency Comparison

| Metric | Binary Approach | Continuous Approach | Improvement |
|--------|----------------|---------------------|-------------|
| **Training Dataset Size** | [PLACEHOLDER: N × 100] | [PLACEHOLDER: N] | 100× reduction |
| **Storage Required** | [PLACEHOLDER: X GB] | [PLACEHOLDER: Y GB] | [PLACEHOLDER: Z%] reduction |
| **Memory Peak Usage** | [PLACEHOLDER: X GB] | [PLACEHOLDER: Y GB] | [PLACEHOLDER: Z%] reduction |

### 5.2 Processing Time Comparison

| Process Stage | Binary Approach | Continuous Approach | Time Saved |
|---------------|----------------|---------------------|------------|
| **Data Preparation** | [PLACEHOLDER: X min] | [PLACEHOLDER: Y min] | [PLACEHOLDER: Z min] |
| **WoE Binning** | [PLACEHOLDER: X min] | [PLACEHOLDER: Y min] | [PLACEHOLDER: Z min] |
| **Model Fitting** | [PLACEHOLDER: X min] | [PLACEHOLDER: Y min] | [PLACEHOLDER: Z min] |
| **Scoring** | [PLACEHOLDER: X min] | [PLACEHOLDER: Y min] | [PLACEHOLDER: Z min] |
| **Total Development Cycle** | [PLACEHOLDER: X hours] | [PLACEHOLDER: Y hours] | [PLACEHOLDER: Z%] faster |

### 5.3 Discriminatory Power Comparison

**Binary Classification Metrics (Binary Approach Only):**
- AUC: [PLACEHOLDER: X.XX]
- KS Statistic: [PLACEHOLDER: X.XX]
- Gini Coefficient: [PLACEHOLDER: X.XX]

**Continuous Metrics (Both Approaches):**

| Metric | Binary Approach | Continuous Approach | Difference |
|--------|----------------|---------------------|------------|
| **Kendall's Tau** | [PLACEHOLDER: X.XX] | [PLACEHOLDER: X.XX] | [PLACEHOLDER: X.XX] |
| **Spearman's Rho** | [PLACEHOLDER: X.XX] | [PLACEHOLDER: X.XX] | [PLACEHOLDER: X.XX] |
| **MSE** | [PLACEHOLDER: X.XX] | [PLACEHOLDER: X.XX] | [PLACEHOLDER: X.XX] |
| **MAE** | [PLACEHOLDER: X.XX] | [PLACEHOLDER: X.XX] | [PLACEHOLDER: X.XX] |
| **RMSE** | [PLACEHOLDER: X.XX] | [PLACEHOLDER: X.XX] | [PLACEHOLDER: X.XX] |

**Interpretation:** [PLACEHOLDER: Discussion of whether discrimination is comparable]

### 5.4 Segmentation Analysis

**Segment-Level Comparison:**

| Segment | Binary Approach Mean LGD | Continuous Approach Mean LGD | Difference | Observations |
|---------|-------------------------|------------------------------|------------|--------------|
| **Very Low Risk** | [PLACEHOLDER: X%] | [PLACEHOLDER: Y%] | [PLACEHOLDER: Z%] | [PLACEHOLDER: N] |
| **Low Risk** | [PLACEHOLDER: X%] | [PLACEHOLDER: Y%] | [PLACEHOLDER: Z%] | [PLACEHOLDER: N] |
| **Medium Risk** | [PLACEHOLDER: X%] | [PLACEHOLDER: Y%] | [PLACEHOLDER: Z%] | [PLACEHOLDER: N] |
| **High Risk** | [PLACEHOLDER: X%] | [PLACEHOLDER: Y%] | [PLACEHOLDER: Z%] | [PLACEHOLDER: N] |
| **Very High Risk** | [PLACEHOLDER: X%] | [PLACEHOLDER: Y%] | [PLACEHOLDER: Z%] | [PLACEHOLDER: N] |

**Segment Separation Quality:**
- Binary Approach KS between segments: [PLACEHOLDER: X.XX]
- Continuous Approach ANOVA F-statistic: [PLACEHOLDER: X.XX]

**Visual Comparison:**
[PLACEHOLDER: Chart showing LGD distribution by segment for both approaches]

**Key Finding:** [PLACEHOLDER: Summary of segmentation consistency]

### 5.5 Predictive Accuracy (Out-of-Time Performance)

**Holdout Period Results:**

| Metric | Binary Approach | Continuous Approach | Better Performance |
|--------|----------------|---------------------|-------------------|
| **Mean Absolute Error** | [PLACEHOLDER: X.XX] | [PLACEHOLDER: Y.XX] | [PLACEHOLDER] |
| **Root Mean Squared Error** | [PLACEHOLDER: X.XX] | [PLACEHOLDER: Y.XX] | [PLACEHOLDER] |
| **Median Absolute Error** | [PLACEHOLDER: X.XX] | [PLACEHOLDER: Y.XX] | [PLACEHOLDER] |
| **R-squared** | [PLACEHOLDER: X.XX] | [PLACEHOLDER: Y.XX] | [PLACEHOLDER] |

**Calibration Quality:**
[PLACEHOLDER: Calibration plots or statistics showing predicted vs. actual LGD]

### 5.6 Model Stability

**Population Stability Index (PSI):**

| Time Period | Binary Approach PSI | Continuous Approach PSI | Assessment |
|-------------|-------------------|------------------------|------------|
| [Period 1] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| [Period 2] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| [Period 3] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

**Interpretation:** PSI < 0.10 = Stable, 0.10-0.25 = Monitor, > 0.25 = Significant shift

### 5.7 Summary Scorecard

| Evaluation Dimension | Binary Approach | Continuous Approach | Winner |
|---------------------|----------------|---------------------|--------|
| **Discriminatory Power** | [PLACEHOLDER: Rating] | [PLACEHOLDER: Rating] | [PLACEHOLDER] |
| **Predictive Accuracy** | [PLACEHOLDER: Rating] | [PLACEHOLDER: Rating] | [PLACEHOLDER] |
| **Segmentation Quality** | [PLACEHOLDER: Rating] | [PLACEHOLDER: Rating] | [PLACEHOLDER] |
| **Processing Efficiency** | [PLACEHOLDER: Rating] | [PLACEHOLDER: Rating] | [PLACEHOLDER] |
| **Memory Efficiency** | [PLACEHOLDER: Rating] | [PLACEHOLDER: Rating] | [PLACEHOLDER] |
| **Model Stability** | [PLACEHOLDER: Rating] | [PLACEHOLDER: Rating] | [PLACEHOLDER] |
| **Governance Simplicity** | [PLACEHOLDER: Rating] | [PLACEHOLDER: Rating] | [PLACEHOLDER] |
| **Team Familiarity** | [PLACEHOLDER: Rating] | [PLACEHOLDER: Rating] | [PLACEHOLDER] |
| **Overall Score** | [PLACEHOLDER: X/8] | [PLACEHOLDER: Y/8] | [PLACEHOLDER] |

---

## 6. Risk Assessment

### 6.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Performance Degradation** | Low | High | Parallel validation against current approach; establish performance thresholds before full migration |
| **Implementation Bugs** | Medium | Medium | Comprehensive unit testing; shadow scoring period |
| **Library Dependency** | Low | Medium | OptBinning is open-source and actively maintained; maintain version control |
| **Integration Issues** | Medium | Low | Phased rollout; maintain dual capabilities during transition |

### 6.2 Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Team Knowledge Gap** | High | Low | Training program; documentation; phased transition |
| **Validation Delays** | Medium | Medium | Early engagement with Model Validation; provide comparative analysis |
| **Production Issues** | Low | High | Extended UAT period; rollback plan |
| **Stakeholder Resistance** | Medium | Medium | Clear communication of benefits; demonstrate equivalence |

### 6.3 Regulatory Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Regulator Questions** | Medium | Medium | Proactive documentation; alignment with Basel advanced approaches |
| **Approval Delays** | Low | Low | Continuous target modeling accepted under AIRB; prepare detailed methodology paper |
| **Increased Scrutiny** | Medium | Low | Transparent validation; demonstrate robustness |

### 6.4 Overall Risk Rating

**Risk Level:** [PLACEHOLDER: Low/Medium/High]

**Risk Tolerance:** The proposed change represents a methodological evolution rather than a fundamental paradigm shift. Models using both approaches produce scorecards with similar structure and interpretability. Risk is further mitigated by:
- Parallel validation during transition
- Ability to revert to binary approach if needed
- Extensive testing and documentation

---

## 7. Implementation Roadmap

### 7.1 Phased Approach

**Phase 1: Proof of Concept (Months 1-2)**
- [ ] Develop prototype continuous model on historical data
- [ ] Complete comparative analysis (this whitepaper)
- [ ] Present findings to Credit Risk leadership
- [ ] Decision: Proceed to pilot or maintain current approach

**Phase 2: Pilot Implementation (Months 3-5)**
- [ ] Select pilot portfolio (e.g., single product segment)
- [ ] Develop production-grade continuous model
- [ ] Parallel run with current binary model
- [ ] Validate results and gather team feedback
- [ ] Engage Model Validation for preliminary review

**Phase 3: Validation & Approval (Months 6-8)**
- [ ] Formal Model Validation review
- [ ] Update Model Risk Management (MRM) documentation
- [ ] Regulatory notification (if required)
- [ ] Governance committee approval
- [ ] Audit preparation

**Phase 4: Production Deployment (Months 9-10)**
- [ ] Production implementation for pilot portfolio
- [ ] Shadow scoring period (both models running)
- [ ] Performance monitoring and comparison
- [ ] Stakeholder training
- [ ] Documentation finalization

**Phase 5: Full Rollout (Months 11-18)**
- [ ] Expand to additional portfolios sequentially
- [ ] Retire binary approach for migrated portfolios
- [ ] Ongoing monitoring and recalibration
- [ ] Lessons learned documentation

### 7.2 Resource Requirements

**Team Allocation:**
- Lead Modeler: [PLACEHOLDER: X% FTE for Y months]
- Supporting Modelers: [PLACEHOLDER: X% FTE for Y months]
- Data Engineering: [PLACEHOLDER: X% FTE for Y months]
- Model Validation: [PLACEHOLDER: X% FTE for Y months]
- Governance/Documentation: [PLACEHOLDER: X% FTE for Y months]

**Technology:**
- OptBinning library (open-source, no licensing cost)
- Python environment (already in place)
- Additional compute resources: [PLACEHOLDER: if needed]

**Budget:**
- Software/Infrastructure: [PLACEHOLDER: $X]
- External Consulting (if needed): [PLACEHOLDER: $X]
- Training: [PLACEHOLDER: $X]
- **Total Estimated Cost:** [PLACEHOLDER: $X]

### 7.3 Success Criteria

**Must-Have (Go/No-Go):**
- [ ] Discriminatory power within 5% of binary approach
- [ ] Segmentation shows consistent LGD ranking
- [ ] Model Validation approval obtained
- [ ] No material increase in computational time for production scoring

**Should-Have:**
- [ ] Development cycle time reduced by >30%
- [ ] Memory usage reduced by >50%
- [ ] Team confident in new methodology
- [ ] Documentation meets regulatory standards

**Nice-to-Have:**
- [ ] Improved predictive accuracy vs. binary approach
- [ ] Positive feedback from regulators
- [ ] Capability to extend to other continuous targets (EAD)

---

## 8. Governance and Validation

### 8.1 Model Documentation Requirements

**Updated Documentation:**
1. **Model Methodology Document**
   - Detailed description of continuous WoE approach
   - Mathematical formulations
   - Comparison to previous approach
   - Rationale for change

2. **Model Development Document**
   - Data sources and preparation
   - Feature engineering process
   - Model selection and testing
   - Performance validation

3. **Model Implementation Document**
   - Scoring algorithm
   - Production architecture
   - Monitoring procedures
   - Contingency plans

4. **Model Validation Report**
   - Independent validation findings
   - Comparative analysis vs. binary approach
   - Limitation and assumptions
   - Recommendations and conditions

### 8.2 Validation Framework

**Conceptual Soundness:**
- [ ] Theoretical foundation of continuous WoE
- [ ] Appropriateness for LGD modeling
- [ ] Alignment with regulatory guidance
- [ ] Literature review and industry practice

**Data Quality:**
- [ ] Data integrity checks
- [ ] Representativeness of development sample
- [ ] Treatment of outliers and missing values
- [ ] Stability of data over time

**Model Development:**
- [ ] Feature selection rationale
- [ ] Model specification testing
- [ ] Coefficient signs and magnitudes
- [ ] Statistical significance
- [ ] Multicollinearity assessment

**Performance:**
- [ ] In-sample fit
- [ ] Out-of-sample performance
- [ ] Out-of-time validation
- [ ] Benchmarking vs. binary approach
- [ ] Segment analysis

**Stability:**
- [ ] PSI over multiple periods
- [ ] Sensitivity analysis
- [ ] Stress testing
- [ ] Backtesting against realized LGD

**Ongoing Monitoring:**
- [ ] Score distribution monitoring
- [ ] Segment migration analysis
- [ ] Challenger model comparison
- [ ] Annual model review process

### 8.3 Model Risk Rating

**Current Binary Approach Risk Rating:** [PLACEHOLDER: Low/Medium/High]

**Proposed Continuous Approach Risk Rating:** [PLACEHOLDER: Low/Medium/High]

**Risk Factors:**
- Model complexity: [PLACEHOLDER: Assessment]
- Data quality: [PLACEHOLDER: Assessment]
- Use criticality: [PLACEHOLDER: Assessment]
- Novel methodology: [PLACEHOLDER: Assessment]

**Recommended Review Frequency:**
- Quarterly monitoring: Yes/No
- Annual validation: Yes
- Ad-hoc review triggers: [PLACEHOLDER: List triggers]

---

## 9. Recommendations

### 9.1 Primary Recommendation

**[PLACEHOLDER: Based on empirical results, provide clear recommendation]**

**Option A: Adopt Continuous Approach**
*Recommended if results show comparable or better performance with significant efficiency gains*

**Rationale:**
- [PLACEHOLDER: Key finding 1]
- [PLACEHOLDER: Key finding 2]
- [PLACEHOLDER: Key finding 3]

**Next Steps:**
1. Secure leadership approval for pilot implementation
2. Initiate Phase 2 of roadmap
3. Allocate resources as outlined in Section 7.2
4. Begin stakeholder communication plan

**Option B: Maintain Binary Approach**
*Recommended if continuous approach shows material performance degradation*

**Rationale:**
- [PLACEHOLDER: Key concern 1]
- [PLACEHOLDER: Key concern 2]

**Next Steps:**
1. Document findings for future reference
2. Monitor OptBinning library development
3. Revisit analysis in [X] years
4. Focus optimization efforts on current approach

**Option C: Hybrid Approach**
*Recommended if results are mixed across portfolios*

**Rationale:**
- [PLACEHOLDER: Portfolio-specific findings]

**Next Steps:**
1. Deploy continuous approach for [specific portfolios]
2. Maintain binary approach for [specific portfolios]
3. Gradual migration as appropriate

### 9.2 Supporting Recommendations

**Regardless of Primary Decision:**

1. **Knowledge Building**
   - Conduct team training on continuous target modeling
   - Stay current with industry developments
   - Participate in industry forums on LGD modeling

2. **Infrastructure Investment**
   - Enhance computational capabilities to support larger datasets
   - Modernize model development platform
   - Improve version control and reproducibility

3. **Methodology Enhancement**
   - Explore additional regression techniques (Tobit, Beta, Quantile)
   - Investigate machine learning comparisons
   - Consider hybrid scorecard-ML approaches

4. **Governance Strengthening**
   - Establish clear criteria for methodology selection
   - Create reusable validation templates
   - Develop stakeholder communication playbooks

---

## 10. Appendices

### Appendix A: Technical Implementation Details

**A.1 Binary Transformation Code Example**

```python
import pandas as pd
import numpy as np
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

def expand_lgd_binary(df, lgd_col='lgd', id_col='account_id'):
    """
    Expand LGD dataset for binary modeling.
    Each record duplicated 100 times with binary target.
    """
    expanded = []
    
    for idx, row in df.iterrows():
        lgd_value = row[lgd_col]
        n_ones = int(round(lgd_value * 100))
        n_zeros = 100 - n_ones
        
        for i in range(100):
            new_row = row.copy()
            new_row['target_binary'] = 1 if i < n_ones else 0
            new_row['original_id'] = row[id_col]
            new_row['duplicate_id'] = i
            expanded.append(new_row)
    
    return pd.DataFrame(expanded)

# Expand data
expanded_df = expand_lgd_binary(train_df)

# Fit binning process
binning = BinningProcess(
    variable_names=predictor_columns,
    categorical_variables=categorical_columns
)
binning.fit(expanded_df[predictor_columns], expanded_df['target_binary'])

# Build scorecard
scorecard = Scorecard(
    binning_process=binning,
    estimator=LogisticRegression(max_iter=1000),
    scaling_method='min_max',
    scaling_method_params={'min': 0, 'max': 1000}
)
scorecard.fit(expanded_df[predictor_columns], expanded_df['target_binary'])

# Score and collapse
expanded_df['score'] = scorecard.predict(expanded_df[predictor_columns])
final_scores = expanded_df.groupby('original_id')['score'].mean()
```

**A.2 Continuous Target Code Example**

```python
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import HuberRegressor

# No data expansion needed - use original dataframe

# Fit binning process for continuous target
binning = BinningProcess(
    variable_names=predictor_columns,
    categorical_variables=categorical_columns
)
binning.fit(train_df[predictor_columns], train_df['lgd'])

# Build scorecard with continuous target
scorecard = Scorecard(
    binning_process=binning,
    estimator=HuberRegressor(epsilon=1.35),  # Robust to outliers
    scaling_method='min_max',
    scaling_method_params={'min': 0, 'max': 1000}
)
scorecard.fit(train_df[predictor_columns], train_df['lgd'])

# Score directly - no averaging needed
scores = scorecard.predict(train_df[predictor_columns])
```

**A.3 Alternative Regression Models for Continuous LGD**

```python
# Tobit Regression (handles censoring at 0 and 1)
from tobit import TobitRegression
tobit_model = TobitRegression(lower=0, upper=1)

# Beta Regression (models values in (0,1) interval)
from statsmodels.discrete.discrete_model import Beta
beta_model = Beta(endog, exog)

# Fractional Logit
from statsmodels.discrete.discrete_model import Logit
# Transform target then use logit

# Quantile Regression (model conditional quantiles)
from sklearn.linear_model import QuantileRegressor
quantile_model = QuantileRegressor(quantile=0.5)
```

### Appendix B: Continuous WoE Calculation Details

**Mathematical Foundation:**

For a continuous target variable Y and predictor X with bins B₁, B₂, ..., Bₖ:

```
WoE(Bᵢ) = |mean(Y | X ∈ Bᵢ) - mean(Y)|
```

Where: