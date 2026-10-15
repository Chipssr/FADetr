# FADetr release checklist

## Prepared locally

- [x] Name the project FADetr and describe the implemented DAFF/AGA path.
- [x] Acknowledge upstream work and distinguish its results from FADetr.
- [x] Remove stale package imports/exports blocking the primary path.
- [x] Ignore data, weights, logs, caches and local credentials.
- [x] Include Apache-2.0 text and a third-party provenance note.
- [x] Add static validation and a GitHub CI workflow.
- [x] Add model-build preflight for the full upstream environment.

## Before claiming reproducible paper results

- [ ] Record exact upstream training-source URL and commit SHA; compare
      licenses, NOTICE files and adapted-file notices against this repository.
- [ ] Publish an environment export from the working training machine, with
      Python, PyTorch, CUDA, MMCV, MMEngine, MMDetection and MMRotate versions.
- [ ] Install into a clean compatible upstream checkout; pass model build,
      initialization, a real forward pass and a short training run.
- [ ] Define disjoint training/evaluation data, official DOTA split, patch size,
      overlap, scale and merge procedure.
- [ ] Reproduce the primary run; record seed, GPUs, effective batch size,
      resolved config, metrics and complete training/evaluation logs.
- [ ] Publish checkpoint, SHA-256 checksum and source config; verify loading
      and inference from a clean checkout.
- [ ] Add controlled DAFF/AGA ablations and the baseline on the same DOTA
      protocol and training budget. Existing DIOR ablations are a separate setup.
- [ ] Measure parameters, FLOPs and speed; record hardware, input size,
      precision, batch size, warm-up, timing and postprocessing details.
- [ ] Add verified paper title, authors, affiliations and citation when ready.
- [ ] Check target venue anonymity policy before linking a public
      author-identifying repository in a submission.

## GitHub publication

- [ ] Confirm GitHub owner and repository name/availability; connect the account.
- [ ] Review exact upload files, create the repository and push source.
- [ ] Check GitHub CI; tag the fixed revision used for the paper results.

No remote repository, accuracy result or trained artifact is created by this list.
