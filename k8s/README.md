# Docker & Kubernetes Deployments

This work can be run locally, but this folder contains a docker image definition and kubernetes set up for experiments
running on a cluster (e.g. Nautilus).

## Persistent Volume Claim (`pvc.yml`)

The `pvc.yml` defines a persistent volume claim for storing data related to this work, useful as a working directory
for any experiments.