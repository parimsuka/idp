# DEM-Based Angle of Repose (AOR) Prediction Using Machine Learning

## Overview

This repository contains the code, data, and documentation for a project that leverages machine learning (ML) techniques to predict the angle of repose (AOR) of pharmaceutical powders using parameters derived from Discrete Element Method (DEM) simulations. By accurately modeling the relationship between microparameters (e.g., friction, cohesion, particle properties) and macroparameters like AOR, this work aims to streamline the DEM calibration process, reduce trial-and-error, and improve simulation accuracy for industrial applications.

## Key Features

- **DEM Calibration Framework:**  
  Integrates sensitivity analysis, correlation studies, and ML modeling to efficiently determine the microparameters that strongly influence AOR.
  
- **Machine Learning Models:**  
  Employs various ML algorithms—such as Support Vector Regression (SVR), Random Forest, Gradient Boosting, XGBoost, and a Stacking Regressor—to compare performance and find the most accurate approach.
  
- **Hyperparameter Tuning:**  
  Utilizes grid search or Bayesian optimization to refine model parameters, ensuring optimal predictive accuracy.
  
- **Generalization to Unseen Data:**  
  Tests and validates models on previously unseen parameter sets, confirming their robustness and applicability to real-world scenarios.
  
- **Focus on Pharmaceutical Powders:**  
  Targets the pharmaceutical industry’s need for reliable process simulations, enabling more informed decision-making in powder handling and dosage form manufacturing.


## References

Relevant literature and references are included in the `docs/` directory and cited within the LaTeX documentation.


---

**Contact:**  
For questions, comments, or collaboration inquiries, please contact me at sukaparim@gmail.com. 
