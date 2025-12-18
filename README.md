# 📈 Algorithmic Trading Strategy Development on Multi-Feature Time Series
### Inter IIT Tech Meet 14.0 — Ebullient Securities

This repository contains the **mid term and end term trading strategies** developed by **IIT Guwahati** for **multi feature time series analysis** as part of the problem statement provided by **Ebullient Securities** for **Inter IIT Tech Meet 14.0**. The work focuses on extracting **predictive signals from high frequency anonymized financial datasets** using systematic quantitative techniques.

---

## 📊 Datasets Overview

Two anonymized time series datasets are provided namely **EBX** and **EBY**, each representing a **distinct yet structurally similar market instrument**.

### Dataset Characteristics

- **1 second interval high frequency data**
- **A core time series** (price or index-like signal)
- **Hundreds of masked features** grouped into categories such as
  - **Price Based**
  - **Volatility Based**
  - **Volume Based**
  - **Alternate Data Based**
  - **Other derived features**

Due to their size, the datasets together **exceed 180 GB** and are therefore **not included directly in this repository**.

---

## 🔗 Dataset Access (Kaggle Links)

### EBY Dataset
https://www.kaggle.com/datasets/interiit/eby-dataset

### EBX Dataset (Distributed in 4 Parts)

The EBX dataset is split into **four parts**. Instructions to merge them are provided inside **EBX_0**.

- https://www.kaggle.com/datasets/interiit/ebx-dataset0
- https://www.kaggle.com/datasets/interiit/ebx-dataset1
- https://www.kaggle.com/datasets/interiit/ebx-dataset2
- https://www.kaggle.com/datasets/interiit/ebx-dataset3

---

## ⏱️ OHLC Data Construction

From the original **1 second interval datasets**, **1 minute and 2 minute OHLC data** for both **EBX** and **EBY** have been constructed and provided in this repository. These aggregated datasets were used extensively in the **End Term Strategy**.

---

## 🧠 Strategy Documentation

- **Detailed explanations** of the strategy logic, feature usage, and modeling approach
- **Clear instructions** on how to run the code and reproduce results

are available in the respective **README.md** and **Report files** inside both the **mid term** and **end term** submission folders.

---
## 📁 Project Structure

```
.
├── EBX_1MIN
├── EBX_2MIN
├── EBY_1MIN
├── EBY_2MIN
├── FINAL-ENDTERM
│   ├── alpha_research
│   ├── EBX.py
│   ├── EBY.py
│   ├── Idea_Summary_Team_33.pdf
│   ├── Performance_Report_Team_33.pdf
│   └── README.md
├── mid_eval
│   ├── FINAL_SUB_X
│   ├── FINAL_SUB_Y
│   ├── alpha_research
│   ├── Backtest_EBX_EBY.py
│   ├── Backtest_results.txt
│   ├── README.md
│   ├── Report.pdf
│   ├── config.json
│   └── requirements.txt
├── EbullientSecurities_H1_TechMeet14.pdf
└── README.md
```
## 📬 Contact

- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/ruchir-sharma-243a10337/)

