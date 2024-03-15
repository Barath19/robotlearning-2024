# Bitbots Gym Environment

![b-it-bot](https://github.com/Barath19/robotlearning-2024/blob/main/docs/images/1024.png)

### Installation

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation.


```bash
git clone https://github.com/Barath19/robotlearning-2024.git
cd robotlearning-2024.git
pip install -e .
```



### Training

```bash
python train.py task=KinovaCabinet 
```
## Testing

To load a trained checkpoint and only perform inference (no training), pass `test=True` 
as an argument, along with the checkpoint name. To avoid rendering overhead, you may 
also want to run with fewer environments using `num_envs=64`:

```bash
python train.py task=KinovaCabinet checkpoint=kinovacabinet.pth test=True num_envs=64
```






