# Bitbots Gym Environment

![b-it-bot](img/1024.png)

### Installation

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. We highly recommend using a conda environment 
to simplify set up.

```bash
pip install -e .
```



### Loading trained models // Checkpoints

Checkpoints are saved in the folder `runs/EXPERIMENT_NAME/nn` where `EXPERIMENT_NAME` 
defaults to the task name, but can also be overridden via the `experiment` argument.

To load a trained checkpoint and continue training, use the `checkpoint` argument:

```bash
python train.py task=KinovaCabinet checkpoint=runs/Ant/nn/Ant.pth
```

To load a trained checkpoint and only perform inference (no training), pass `test=True` 
as an argument, along with the checkpoint name. To avoid rendering overhead, you may 
also want to run with fewer environments using `num_envs=64`:

```bash
python train.py task=KinovaCabinet checkpoint=kinovacabinet.pth test=True num_envs=64
```






