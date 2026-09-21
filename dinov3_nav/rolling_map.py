"""Metric rolling LiDAR map with optional DINO visual traversability."""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import List, Tuple
import numpy as np
from .bev import BEVConfig, BEVGrid

@dataclass
class RollingMapConfig:
    ttl_s: float = 2.0
    ground_tolerance_m: float = .12
    obstacle_height_m: float = .18
    min_points_per_cell: int = 1
    visual_unknown_value: float = .55
    self_filter_radius_m: float = .45

class RollingLocalMap:
    """Stores LiDAR observations in odom and re-rasterizes them in base_link."""
    def __init__(self, bev_cfg: BEVConfig, cfg: RollingMapConfig):
        self.bev_cfg, self.cfg, self._chunks = bev_cfg, cfg, []

    def add(self, points_base: np.ndarray, visual_traversability: np.ndarray,
            T_odom_from_base: np.ndarray, stamp_s: float) -> None:
        # Reject returns from robot body, LiDAR housing and mounting hardware.
        # They otherwise become a persistent obstacle under base_link and make
        # every footprint rollout collide at its first integration step.
        keep_self = np.hypot(points_base[:, 0], points_base[:, 1]) >= self.cfg.self_filter_radius_m
        points_base = points_base[keep_self]
        visual_traversability = visual_traversability[keep_self]
        if points_base.size == 0: return
        z = points_base[:, 2]
        ground_z = float(np.percentile(z, 15))
        obstacle = z > ground_z + self.cfg.obstacle_height_m
        ground = z <= ground_z + self.cfg.ground_tolerance_m
        keep = obstacle | ground
        p = points_base[keep]
        if not len(p): return
        xy_odom = p[:, :2] @ T_odom_from_base[:2, :2].T + T_odom_from_base[:2, 3]
        self._chunks.append((stamp_s, xy_odom.astype(np.float32), obstacle[keep],
                             visual_traversability[keep].astype(np.float32)))
        self._chunks = [c for c in self._chunks if stamp_s-c[0] <= self.cfg.ttl_s]

    def rasterize(self, T_odom_from_base: np.ndarray, stamp_s: float) -> BEVGrid:
        cfg = self.bev_cfg; nx = int(ceil((cfg.x_max-cfg.x_min)/cfg.resolution)); ny = int(ceil((cfg.y_max-cfg.y_min)/cfg.resolution))
        obs = np.zeros((nx,ny), np.float32); trav = np.zeros((nx,ny), np.float32); obst = np.zeros((nx,ny), np.int32)
        T_base_from_odom = np.linalg.inv(T_odom_from_base)
        self._chunks = [c for c in self._chunks if stamp_s-c[0] <= self.cfg.ttl_s]
        for ts, xy, is_obst, visual in self._chunks:
            current = xy @ T_base_from_odom[:2,:2].T + T_base_from_odom[:2,3]
            i=((current[:,0]-cfg.x_min)/cfg.resolution).astype(np.int32); j=((current[:,1]-cfg.y_min)/cfg.resolution).astype(np.int32)
            inside=(i>=0)&(i<nx)&(j>=0)&(j<ny)
            if not inside.any(): continue
            decay=np.exp(-(stamp_s-ts)/max(self.cfg.ttl_s*.5,.1))
            i,j=i[inside],j[inside]; v=visual[inside]; o=is_obst[inside]
            np.add.at(obs,(i,j),decay); np.add.at(trav,(i,j),decay*v); np.add.at(obst,(i,j),o.astype(np.int32))
        # Decay expresses confidence, not whether a real LiDAR hit occurred.
        observed=obs>=max(.05, .25*self.cfg.min_points_per_cell)
        traversability=np.full((nx,ny), self.cfg.visual_unknown_value, np.float32)
        np.divide(trav,np.maximum(obs,1e-6),out=traversability,where=obs>0)
        obstacle=obst>0; traversability[obstacle]=0.0; observed[obstacle]=True
        count=np.rint(obs).astype(np.int32)
        return BEVGrid(traversability,observed,obstacle,count,np.rint(trav).astype(np.int32),obstacle_count=obst,cfg=cfg)
