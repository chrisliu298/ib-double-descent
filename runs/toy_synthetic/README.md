# Toy/Synthetic setup

Setting up a toy-synthetic experiment based on 
[discussion on Discord](https://discord.com/channels/1022698024817938452/1028058669344108644/1044794251990540371):
> - Random feature networks with layer dimensions 784xWx2, where W is the network width in [4, 400] with step size 4
> - Binary MNIST and Fashion-MNIST with only 100 training examples
> - SGD optimizer
> - Cross-entropy loss and MSE loss
> - Initial learning rate 1e-2 with inverse square root decay per epoch (inverse_sqrt)
> - Train for 1000 epochs in total
> - Batch size 1 (I use 1 because it has the fastest convergence rate, and convergence to 0 training error is crucial to produce double descent. Larger batch sizes might also work if convergence is ensured.)
> - Label noise 0.05 (Later I'll explain why.)

Outer file `toy_synthetic.json` holds a **template** for each run.

`generate_run_files.py` populates each run in `definitions` folder.

All can be run with `TODO`