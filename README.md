# Auto-Encoder

Repo to play with (auto) encoding and feature spaces.

## CNN-Bases Auto-Endcoder `auto_encoder.py`

A CNN-Based encoder-decoder net, where we can tell the latent dim as a parameter.
It trains then with the MNIST data set.

## Variational Auto-Encoder `vae_linear.py`

A very simple auto-encoder with just three respectively two linear (fully-connected) nets.
More important is the part inbetween, since it is a variational auto-encoder.
The `VariationalEncoder` has actually two "parallel" linear layers on the second layer.
One, we interpret as the mean (mu), the other as the standard deviation (sigma).
The final encoding is then not the deterministic latent space, but a sample drawn from the learnt distribution.
Deeper explanations are in the code.

