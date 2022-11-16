# Docker & Kubernetes Deployments

This work can be run locally, but this folder contains a docker image definition and kubernetes set up for experiments
running on a cluster (e.g. Nautilus).

## Persistent Volume Claim (`pvc.yml`)

The `pvc.yml` defines a persistent volume claim for storing data related to this work, useful as a working directory
for any experiments.

This volume has been added to the `jlab-nlp` Namespace, but could be added to others as needed.

## Dockerfile

Dockerfile defines a container in which the repository is copied, as well as all project dependencies installed. 
**NOTE: the base image is defined with a user account `ibdd`. 
It has `sshd` running as this non-privileged user, with my SSH key as the only allowed key.** 
TODO: if we need to use ssh directly (PyCharm remote debugging, etc.), we should create a shared SSH key

Another warning: the GH Action is defined to save cycles by not building unless the container definition changes or more
dependencies are added. The repo is installed in edit mode, but to run a job with fresh code the entry-point should 
clone the repository into place. See `debug.yml` for an example.