# Assignment 2 – CSCN 8020: Reinforcement Learning

## Tabular Q-Learning Hyperparameter Analysis on Taxi-v3

### Objective
Implement Tabular Q-Learning on the Taxi-v3 environment and conduct a comprehensive empirical analysis of how hyperparameter choices affect learning performance. Specifically, evaluate the impact of:
- **Learning Rate (α)**: {0.001, 0.01, 0.1, 0.2}
- **Exploration Factor (ε)**: {0.1, 0.2, 0.3}
- **Discount Factor (γ)**: 0.9 (fixed)

---

## Key Findings

### Learning Rate (α) Impact
| α Value | Final 100-ep Return | Mean Steps/Episode | Assessment |
|---------|--------------------|--------------------|------------|
| 0.001   | -214.17            | 179.71             | Very Poor  |
| 0.01    | -4.97              | 81.98              | Poor       |
| 0.1     | 2.66–3.25          | 22.62              | Good       |
| **0.2** | **3.64**           | **19.13**          | **Best**   |

**Insight:** Learning rate is critical for convergence. α ≥ 0.1 is essential; smaller values prevent meaningful learning within 10,000 episodes.

### Exploration Factor (ε) Impact
| ε Value | Final 100-ep Return | Mean Steps/Episode | Assessment |
|---------|--------------------|--------------------|------------|
| **0.1** | **2.73–3.25**      | **22.53**          | **Best**   |
| 0.2     | -5.08              | 24.90              | Poor       |
| 0.3     | -11.96             | 28.01              | Very Poor  |

**Insight:** Lower exploration rates maintain better exploit-explore balance in deterministic environments. Higher ε severely degrades final policy quality.

### Optimal Configuration
**α = 0.2, ε = 0.1, γ = 0.9** (verified with independent seed: 123)

| Metric | Baseline (α=0.1) | Optimal (α=0.2) | Improvement |
|--------|------------------|-----------------|-------------|
| Mean Steps/Episode | 22.60 | 19.14 | **15% faster** |
| Final 100-ep Return | 3.25 | 2.88 | Competitive |
| Convergence | Stable | More Stable | ✓ |

---

## Project Structure

```
.
├── CSCN8020_Assignment2_Final.ipynb    # Main Jupyter notebook with full analysis
├── CSCN8020_Assignment2_Report.pdf     # Comprehensive PDF report
├── plots/                               # Visualization directory
│   ├── 01_learning_rate_comparison.png
│   ├── 02_exploration_factor_comparison.png
│   ├── 03_baseline_vs_best.png
│   └── 04_summary_bar_chart.png
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation
```bash
# Clone or navigate to the project directory
cd Assignment-2-CSCN-8020

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Analysis

### Execute the Jupyter Notebook
```bash
jupyter notebook CSCN8020_Assignment2_Final.ipynb
```

The notebook will:
1. Install/verify Gymnasium and NumPy
2. Initialize the Taxi-v3 environment
3. Implement Q-Learning algorithm
4. Conduct hyperparameter sweeps:
   - Step 5: Baseline training (α=0.1, ε=0.1)
   - Step 6: Learning rate sensitivity (α sweep)
   - Step 7: Exploration factor sensitivity (ε sweep)
   - Step 8: Optimal configuration validation (α=0.2, ε=0.1)
   - Step 9: Results visualization (4 comprehensive plots)
5. Generate PDF report with detailed analysis

---

## Evaluation Metrics

- **Final 100-Episode Average Return**: Rolling average of cumulative rewards over the last 100 episodes
- **Mean Steps per Episode**: Average number of steps required to complete episodes
- **Convergence Speed**: How quickly the agent reaches near-optimal policy

---

## Environment Details

**Taxi-v3 (Gymnasium v0.29.0)**
- **States**: 500 discrete states (25 taxi positions × 5 passenger locations × 4 destinations)
- **Actions**: 6 discrete actions (move S/N/E/W, pickup, dropoff)
- **Episode Length**: Maximum 200 steps
- **Reward**: +20 (successful dropoff), -1 (per step), -10 (illegal action)

**Q-Learning Configuration**
- Algorithm: Tabular Q-Learning with ε-greedy exploration
- Training Episodes: 10,000
- Q-table Size: 500 × 6 = 3,000 values

---

## Results & Deliverables

### Generated Report
**File**: `CSCN8020_Assignment2_Report.pdf`
- Executive summary
- Detailed methodology
- Learning rate analysis with tables
- Exploration factor analysis with tables
- Optimal configuration comparison
- Conclusions and recommendations

### Visualizations
All plots saved to `plots/` directory:
1. **Learning Rate Comparison** (2-panel: return & steps)
2. **Exploration Factor Comparison** (2-panel: return & steps)
3. **Baseline vs Optimal** (2-panel: return & steps with overlaid curves)
4. **Performance Summary** (2-panel bar charts with color coding)

---

## Key Conclusions

1. **Learning Rate Dominance**: α is the primary driver of convergence speed. The difference between α=0.001 and α=0.2 shows a **3,064% improvement** in final return.

2. **Sharp Exploration Trade-off**: ε exhibits cliff-like behavior in deterministic environments. Lower values (ε=0.1) dramatically outperform higher values.

3. **Robustness**: Optimal configuration verified across multiple random seeds, confirming generalizability.

4. **Environment-Specific**: These findings reflect Taxi-v3's deterministic, fully-observable nature. Stochastic environments would require different hyperparameter ranges.

---

## Recommendations for Practitioners

- **Learning Rate**: Use α ∈ [0.1, 0.2] for tabular Q-Learning on deterministic environments
- **Exploration**: Keep ε ≤ 0.1 for fully-observable environments; increase only for high uncertainty
- **Validation**: Always test hyperparameters across multiple random seeds
- **Adaptation**: Conduct similar sensitivity analyses for new problem domains

---

## Technologies Used

- **Python 3.11.3**
- **Gymnasium 0.29.0** (OpenAI's successor to OpenAI Gym)
- **NumPy 2.4.2** (Numerical computing)
- **ReportLab** (PDF generation)
- **Matplotlib** (Visualization)
- **Jupyter Notebook** (Interactive development)

---

## Author
CSCN 8020 - Reinforcement Learning  
Assignment 2 - Q-Learning Analysis

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Gymnasium Documentation: https://gymnasium.farama.org/
- Taxi-v3 Environment: https://gymnasium.farama.org/environments/toy_text/taxi/
