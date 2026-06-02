# To-Do
## Classifiaction issue
- Check code for bugs
- Remove covariance constraint
- Play with $\beta$ and number of supervised data points
- Increase window size
- Choose better features
- Try scaling the training data
  - Not scaling but mapping, i.e. not dividing by sigma
- **Feature issue**
  - Don't use windows?
  - ~Move windows one point at a time~ -> already the case
  - Dimensionality reduction after featurization?
- ~Plot x1, x2 instead of PCs to see if it is well separated~
- ~Trick for SPD(semi positive definite) -> $\frac{1}{2}(A + A^{\top})$~ -> Doesn't change anything

## Next
- Plot uncertainty on new data

## Practical implementation for writing
- When demoing, select some points auto and save with known segment from symbolic planner