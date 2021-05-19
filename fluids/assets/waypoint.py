import numpy as np
import pygame
import shapely.geometry

import scipy.interpolate as si
from fluids.assets.shape import Shape
from fluids.assets.waypoint_edge import WaypointEdge


def plan(x0, y0, a0, x1, y1, a1, smooth_level=3000):
    def interpolate(p0, p1, p2, p3, t):
        return [
            p0[0] * 1.0 * ((1 - t) ** 3)
            + p1[0] * 3.0 * t * (1 - t) ** 2
            + p2[0] * 3.0 * (t ** 2) * (1 - t)
            + p3[0] * 1.0 * (t ** 3),
            p0[1] * 1.0 * ((1 - t) ** 3)
            + p1[1] * 3.0 * t * (1 - t) ** 2
            + p2[1] * 3.0 * (t ** 2) * (1 - t)
            + p3[1] * 1.0 * (t ** 3),
        ]

    distance_between_points = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    p0 = [x0, y0]
    p1 = [
        x0 + 0.3 * distance_between_points * np.cos(a0),
        y0 - 0.3 * distance_between_points * np.sin(a0),
    ]
    p2 = [
        x1 - 0.3 * distance_between_points * np.cos(a1),
        y1 + 0.3 * distance_between_points * np.sin(a1),
    ]
    p3 = [x1, y1]

    first_point = interpolate(p0, p1, p2, p3, 0)
    res_path = [first_point]
    for t in np.arange(0, 1, 0.001):
        new_point = interpolate(p0, p1, p2, p3, t)
        old_point = res_path[-1]
        if (new_point[0] - old_point[0]) ** 2 + (
            new_point[1] - old_point[1]
        ) ** 2 > smooth_level:
            res_path.append(new_point)

    res_path.append([x1, y1])
    if a1 - a0 > np.pi:
        a1 = a1 - (2 * np.pi)
    if a0 - a1 > np.pi:
        a0 = a0 - (2 * np.pi)
    new_angles = np.linspace(a0, a1, len(res_path))
    return res_path, new_angles


class Waypoint(Shape):
    def __init__(self, x, y, owner=None, angle=0, nxt=None, **kwargs):

        self.radius = 0
        self.nxt = nxt if nxt else []
        self.owner = owner
        points = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)]

        super(Waypoint, self).__init__(
            x=x, y=y, angle=angle % (2 * (np.pi)), xdim=5, color=(0, 255, 255), **kwargs
        )

    def smoothen(self, smooth_level=3000):
        all_news = []
        new_nxt = []
        for n_p in self.nxt:
            interp = []
            path, angles = plan(
                self.x,
                self.y,
                self.angle % (2 * np.pi),
                n_p.x,
                n_p.y,
                n_p.angle % (2 * np.pi),
                smooth_level=smooth_level,
            )
            new_point = Waypoint(
                path[1][0],
                path[1][1],
                ydim=self.ydim,
                angle=angles[1],
                owner=self.owner,
            )
            all_news.append(new_point)
            new_nxt.append(new_point)
            interp.append(new_point)
            for i in range(2, len(path) - 1):
                next_p = Waypoint(
                    path[i][0],
                    path[i][1],
                    ydim=self.ydim,
                    angle=angles[i],
                    owner=self.owner,
                )
                all_news.append(next_p)
                interp.append(next_p)
                new_point.nxt = [next_p]
                new_point = next_p
            new_point.nxt = [n_p]

        self.nxt = new_nxt
        return all_news

    def create_edges(self, **kwargs):
        new_nxt = []
        for n_p in self.nxt:
            new_nxt.append(WaypointEdge(self, n_p, **kwargs))
        self.nxt = new_nxt

    def render(self, surface, **kwargs):
        kwargs["border"] = None
        super(Waypoint, self).render(surface, **kwargs)
        if "color" in kwargs:
            color = kwargs["color"]
        else:
            color = self.color
        if "nxt" in self.__dict__:
            for next_point in self.nxt:
                pygame.draw.line(
                    surface,
                    color,
                    (int(self.x), int(self.y)),
                    (int(next_point.out_p.x), int(next_point.out_p.y)),
                    1,
                )
