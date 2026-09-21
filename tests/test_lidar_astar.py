"""ROS-free checks for LiDAR rolling geometry and deterministic A* detours."""
import numpy as np
from dinov3_nav.astar import AStarLocalPlanner
from dinov3_nav.bev import BEVConfig
from dinov3_nav.footprint import FootprintConfig
from dinov3_nav.planning_bev import PlanningCostConfig, build_planning_bev
from dinov3_nav.rolling_map import RollingLocalMap, RollingMapConfig

CFG=BEVConfig(resolution=.1,x_min=-.6,x_max=3.0,y_min=-1.5,y_max=1.5)

def test_lidar_geometry_persists_outside_camera_fov():
    memory=RollingLocalMap(CFG,RollingMapConfig(ttl_s=2.0))
    # visual=.55 means camera did not see this point; obstacle must remain hard.
    memory.add(np.array([[1., 1., 0.],[1., 0., .5]],np.float32),np.array([.55,.55],np.float32),np.eye(4),0.)
    grid=memory.rasterize(np.eye(4),.1)
    assert grid.obstacle[grid.xy_to_ij(1.,0.)]
    assert grid.observed[grid.xy_to_ij(1.,1.)]

def test_astar_selects_detour_not_straight_through_wall():
    memory=RollingLocalMap(CFG,RollingMapConfig())
    points=[]
    # Ground returns establish the local ground plane; elevated returns form
    # the wall that A* must route around.
    for x in np.arange(0.,2.8,.2): points.append([x,1.2,0.])
    for y in np.arange(-.4,.41,.1): points.append([1.,y,.5])
    memory.add(np.asarray(points,np.float32),np.ones(len(points),np.float32),np.eye(4),0.)
    grid=memory.rasterize(np.eye(4),0.)
    planning=build_planning_bev(grid,FootprintConfig(length=.4,width=.3,safety_margin=.1),PlanningCostConfig())
    path=AStarLocalPlanner().plan(planning,(2.2,0.))
    assert path.points and any(abs(y)>.45 for _,y in path.points), path
