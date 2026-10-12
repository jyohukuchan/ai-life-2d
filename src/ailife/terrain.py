"""Small static obstacles with connected floor, generated once on the CPU."""
from __future__ import annotations

import numpy as np


def make_obstacles(size: int, density: float, seed: int, width: int | None = None) -> np.ndarray:
    width = size if width is None else width
    obstacles = np.zeros((size, width), dtype=bool)
    remaining = round((size - 2) * (width - 2) * density)
    if not remaining:
        return obstacles

    rng = np.random.default_rng(seed)
    # A free outer floor ring and one free row/column between regions keep
    # every floor tile reachable. Each region contains at most one solid block.
    regions = [(y, x, min(5, size - 2 - y), min(5, width - 2 - x))
               for y in range(2, size - 2, 6)
               for x in range(2, width - 2, 6)]
    rng.shuffle(regions)
    available = sum(min(4, height) * min(4, width) for _, _, height, width in regions)
    for y, x, height, width in regions:
        max_area = min(4, height) * min(4, width)
        available -= max_area
        required = max(1, remaining - available)
        shapes = [(h, w) for h in range(1, min(4, height) + 1)
                  for w in range(1, min(4, width) + 1) if h * w >= required]
        if not shapes:
            raise ValueError("too many obstacles for this map size")
        h, w = shapes[int(rng.integers(len(shapes)))]
        y += int(rng.integers(height - h + 1))
        x += int(rng.integers(width - w + 1))
        cells = [(y + dy, x + dx) for dy in range(h) for dx in range(w)]
        # A truncated final rectangle still has no holes or enclosed floor.
        for yy, xx in cells[:remaining]:
            obstacles[yy, xx] = True
        remaining -= min(remaining, h * w)
        if not remaining:
            break
    if remaining:
        raise ValueError("too many obstacles for this map size")
    return obstacles


def fertility_field(height, width, seed, kind, x_min=0., x_max=1.):
    """The full-square path preserves the original generator exactly."""
    rng=np.random.default_rng(seed)
    if kind=='uniform':return np.ones((height,width),dtype=np.float32)*.2
    cy,cx=np.mgrid[:height,:width]
    field=np.zeros((height,width),dtype=np.float32)
    lo=max(1.,x_min*width);hi=min(width-2.,x_max*width)
    if hi<=lo:raise ValueError('food region is too narrow')
    sigma=max(1.,min(height,(x_max-x_min)*width)*.13)
    for _ in range(3):
        if height==width and x_min==0 and x_max==1:a,b=rng.uniform(1,height-2,2)
        else:a,b=rng.uniform([1.,lo],[height-2.,hi])
        field+=np.exp(-((cy-a)**2+(cx-b)**2)/(2*sigma**2))
    zone=(cx>=x_min*width)&(cx<x_max*width)
    mean=float(field[zone].mean())
    return np.minimum(1,field*(.2/max(mean,1e-6)))


def preserve_food_total(field, target, cap):
    """Redistribute the old probability budget without saturating probabilities."""
    weights=field.astype(np.float64)
    if target<0 or cap<=0 or not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError('invalid food probability budget')
    if target==0:return np.zeros_like(field)
    if target>np.count_nonzero(weights)*cap+1e-6:
        raise ValueError('food probability budget does not fit the food region')
    lo,hi=0.,max(1.,target/max(float(weights.sum()),1e-12))
    while np.minimum(weights*hi,cap).sum()<target:hi*=2
    for _ in range(64):
        mid=(lo+hi)/2
        if np.minimum(weights*mid,cap).sum()<target:lo=mid
        else:hi=mid
    return np.minimum(weights*hi,cap).astype(np.float32)
