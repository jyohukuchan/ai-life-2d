from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import yaml


@dataclasses.dataclass
class Config:
    worlds: int = 4
    initial: int = 32
    capacity: int = 128
    size: int = 0
    height: int = 0
    width: int = 0
    hidden: int = 64
    encoder_width: int = 32
    mode: str = "L3"
    seed: int = 42
    ticks: int = 9000
    radius: int = 5
    symbols: int = 4
    sound_radius: int = 10
    interoception: bool = True
    compass: str = "none"
    founder_energy: float = 50.0
    child_energy: float = 30.0
    birth_overhead: float = 10.0
    energy_max: float = 100.0
    metabolism: float = 0.02
    move_cost: float = 0.01
    food_energy: float = 10.0
    regrowth: float = 0.0001
    initial_food: float = 0.12
    fertility: str = "patches"
    obstacle_density: float = 0.04
    food_x_min: float = 0.0
    food_x_max: float = 1.0
    spawn_x_min: float = 0.0
    spawn_x_max: float = 1.0
    remote_births: bool = False
    food_budget: str = "local"
    food_reference_size: int = 0
    log_reward_components: bool = False
    reward_log_block: int = 1000
    lifespan: int = 3000
    maturity_ratio: float = 0.1
    reproduction: str = "both"
    mutation: float = 0.05
    publish_seconds: float = 0.5
    log_ticks: int = 1000
    checkpoint_ticks: int = 3000
    replay_count: int = 3
    replay_length: int = 256
    observer_seed: int = 72011
    threads: int = 4
    elite_count: int = 0
    fast_forward_extinct: bool = False
    precision: str = "fp32"
    execution: str = "eager"
    fp16_loss_scale: float = 128.0

    def __post_init__(self):
        if not isinstance(self.encoder_width,int) or isinstance(self.encoder_width,bool) or self.encoder_width<1:
            raise ValueError("encoder_width must be a positive integer")
        if self.precision not in {"fp32","fp16","bf16"}:raise ValueError("invalid precision")
        if self.execution not in {"eager","kernels","compiled"}:raise ValueError("invalid execution backend")
        if not math.isfinite(self.fp16_loss_scale) or self.fp16_loss_scale<=0:raise ValueError("invalid fp16 loss scale")
        if not self.size:
            self.size = max(7, round(math.sqrt(100 * self.initial)))
        if bool(self.height)!=bool(self.width):
            raise ValueError("specify both height and width")
        if not self.height:self.height=self.width=self.size
        if min(self.height,self.width)<7:raise ValueError("map dimensions must be at least 7")
        if self.compass not in {"none","masked","absolute"}:
            raise ValueError("compass must be none, masked or absolute")
        if self.mode not in {"L0", "L1", "L2", "L3", "RANDOM"}:
            raise ValueError("mode must be L0, L1, L2, L3 or RANDOM")
        if self.worlds < 1 or not 1 <= self.initial <= self.capacity:
            raise ValueError("invalid world/population counts")
        if self.capacity > (self.height - 2)*(self.width-2):
            raise ValueError("capacity exceeds the walkable map")
        if not 0 <= self.obstacle_density <= 0.2:
            raise ValueError("obstacle_density must be between 0 and 0.2")
        interior=(self.height-2)*(self.width-2)
        if self.capacity > interior-round(interior*self.obstacle_density):
            raise ValueError("capacity exceeds the map's unobstructed cells")
        if self.hidden < 1 or self.radius < 0 or self.symbols < 0:
            raise ValueError("invalid model or sensor dimensions")
        if self.reproduction not in {"both", "clone", "sexual"}:
            raise ValueError("invalid reproduction condition")
        if self.fertility not in {"patches", "uniform"}:
            raise ValueError("invalid fertility condition")
        if not 0 < self.child_energy <= self.energy_max or self.birth_overhead < 0:
            raise ValueError("invalid birth energy accounting")
        if min(self.regrowth, self.metabolism, self.move_cost, self.mutation) < 0:
            raise ValueError("negative physical rate")
        if self.ticks<1 or self.log_ticks<1 or self.checkpoint_ticks<0 or self.threads<1:
            raise ValueError("invalid execution intervals")
        if self.replay_count<0 or self.replay_length<1 or self.publish_seconds<0:
            raise ValueError("invalid observation intervals")
        if self.lifespan<1 or not 0<=self.maturity_ratio<=1 or not 0<=self.initial_food<=1:
            raise ValueError("invalid ecology configuration")
        if not 0<self.founder_energy<=self.energy_max or self.food_energy<0:
            raise ValueError("invalid energy configuration")
        for lo,hi in [(self.food_x_min,self.food_x_max),(self.spawn_x_min,self.spawn_x_max)]:
            if not 0<=lo<hi<=1:raise ValueError("invalid horizontal region")
        if self.food_budget not in {'local','preserve_total'}:raise ValueError("invalid food budget")
        if self.food_reference_size and self.food_reference_size<7:raise ValueError("invalid reference map")
        if not 0<=self.elite_count<=self.capacity:raise ValueError("invalid elite count")
        if self.elite_count and self.mode=="RANDOM":raise ValueError("elite weights require a neural policy")
        if self.reward_log_block<1:raise ValueError("invalid reward log block")

    @property
    def maturity(self):
        return max(1, round(self.lifespan * self.maturity_ratio))

    @property
    def action_sizes(self):
        return (4, 2, self.symbols + 1, 3)

    @property
    def sensory_size(self):
        return (2 * self.radius + 1) ** 2 * 3 + self.symbols + int(self.interoception)

    @property
    def compass_size(self):
        return 0 if self.compass=="none" else 4

    @property
    def observation_size(self):
        return self.sensory_size+self.compass_size

    @property
    def action_size(self):
        return sum(self.action_sizes)

    def as_dict(self):
        return dataclasses.asdict(self)

    def digest(self):
        payload = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    @classmethod
    def load(cls, path):
        with Path(path).open() as f:
            return cls(**(yaml.safe_load(f) or {}))

    @classmethod
    def from_saved(cls, values):
        """Checkpoints predating obstacles describe an unobstructed world."""
        return cls(**{"obstacle_density":0.0, **values})
