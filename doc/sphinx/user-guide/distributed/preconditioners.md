# Distributed Preconditioners

- AMG
- (non-overlapping) Schwarz

## AMG

- supported by PGM
- same as for sequential
- need to provide distributed preconditioners as smoothers
- distributed solver as coarse grid

## Schwarz

- build with
  - local solver
  - generated local solver
- applies solver to local matrix

