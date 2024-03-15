
from .franka_cabinet import FrankaCabinet
from .kinova_cabinet import KinovaCabinet
from .ur5_cabinet import UR5Cabinet
from .hsr_cabinet import HSRCabinet



# Mappings from strings to environments
isaacgym_task_map = {
    "FrankaCabinet": FrankaCabinet,
    "KinovaCabinet": KinovaCabinet,
    "UR5Cabinet": UR5Cabinet,
    "HSRCabinet": HSRCabinet
}
