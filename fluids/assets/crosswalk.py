import numpy as np
from fluids.assets.shape import Shape
from fluids.assets.waypoint import Waypoint


class CrossWalk(Shape):
    def __init__(self, start_wps=[], end_wps=[], **kwargs):
        Shape.__init__(self, color=(0xF1, 0xF4, 0xF5), **kwargs)
        point0 = (self.points[2] + self.points[3]) / 2
        point1 = (self.points[0] + self.points[1]) / 2

        if len(start_wps) and len(end_wps):
            self.start_waypoints = start_wps
            self.end_waypoints = end_wps
        else:
            self.start_waypoints = [
                Waypoint(point0[0], point0[1], owner=self, ydim=5, angle=self.angle),
                Waypoint(
                    point1[0], point1[1], owner=self, ydim=5, angle=self.angle + np.pi
                ),
            ]
            self.end_waypoints = [
                Waypoint(point1[0], point1[1], owner=self, ydim=5, angle=self.angle),
                Waypoint(
                    point0[0], point0[1], owner=self, ydim=5, angle=self.angle + np.pi
                ),
            ]
            self.start_waypoints[0].nxt = [self.end_waypoints[0]]
            self.start_waypoints[1].nxt = [self.end_waypoints[1]]
