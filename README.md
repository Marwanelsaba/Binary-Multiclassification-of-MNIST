# MNIST Manual Project (Fast Manual Version)

This version keeps the project manual while reducing execution time.

## What was optimized
- Logistic Regression is still implemented from scratch, but the math inside each batch uses NumPy arrays instead of slow feature-by-feature Python loops.
- KNN is still implemented from scratch, but distance computation uses handwritten formulas with fast NumPy array operations.
- PCA is still implemented from scratch, but it now uses the dual-covariance trick for the common MNIST case where samples are fewer than pixels. This is the main speed fix.
- Phase 2 hyperparameter tuning was reduced to a smaller manual search grid so the improved pipeline still runs in a practical time.

## Fast run commands

### Phase 1 (all features + all models)
```bash
python main.py --phase 1 --config phase1_fast_config.json --mnist-loader tensorflow
```

### Phase 2 (improved pipeline)
```bash
python main.py --phase 2 --improved --config phase2_fast_config.json --mnist-loader tensorflow
```

## Notes
- Phase 1 fast config uses 300 samples per class for digits 3 and 8.
- Phase 2 fast config uses 40 samples per class for all digits to keep runtime practical.
- Outputs are saved inside `outputs/`.
