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

## Jobs

For now, I've created a `test_job.yml`, which sets things up and runs `src/mi_estimation/tran.py`, with some default
arguments. We can create new kinds of jobs which do this for a batch of runs, etc. To run this, tweak arguments in 
`test_job.yml` and run `kubectl create -f k8s/test_job.yml`

## Debugging in k8s with `debug.yml`

This is a pod definition using the above image, which can be used for remote debugging with an SSH interpreter. To set
this up, one needs to:

1. launch the pod: `kubectl apply -f k8s/debug.yml`
2. Run port forwarding so that you can ssh to your local port to get on the container:
`kubectl port-forward pods/bking2-recoverable-dst 2022:2022`. Note: this command doesn't return, good to run in a 
`screen`.
3. Set up a
[pycharm remote interpreter](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html) to
point to host=`localhost`, port=`2022`.

Again, without building your own base image and re-building the docker image above, you'll end up with an image that
serves this for a user named `idbb` and a my personal SSH key as the only working public key. Later work may make this
more flexible, but for now, you would need to fork and modify 
[kingb12/docker-dev-env](https://github.com/kingb12/docker-dev-env) to add a new key.
