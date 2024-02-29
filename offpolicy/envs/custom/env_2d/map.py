"""
Env 2D
"""

from shapely.geometry import Polygon, Point, box

import random


class Map:
    def __init__(self, width=51, height=51):
        self.x_range = width
        self.y_range = height

        self.obstacles = self.obs_map()
        self.obs_bounds = self.get_obs_bounds()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range

        obstacles = set()
   
        # left boundary
        obstacles.add(box(minx=-1, miny=0, maxx=0, maxy=y))
        # bottom boundary
        obstacles.add(box(minx=0, miny=-1, maxx=x, maxy=0))
        # right boundary
        obstacles.add(box(minx=x, miny=0, maxx=x+1, maxy=y))
        # top boundary
        obstacles.add(box(minx=0, miny=y, maxx=x, maxy=y+1))

        # obstacle_1
        obstacle_1 = box(minx=x//3, miny=0, maxx=x//3+1, maxy=y//2)
        obstacles.add(obstacle_1)

        # obstacle_2
        obstacle_2 = box(minx=x//3*2, miny=y//2, maxx=x//3*2+1, maxy=y)
        obstacles.add(obstacle_2)
     
        return obstacles

    def move(self, point, motion):
        return (point[0] + motion[0], point[1] + motion[1])
    
    def is_collision(self, polygon:Polygon):
        for obs in self.obstacles:
            if polygon.intersects(obs):
                return True

        return False

    def random_point(self):
        while True:
            point = (random.randint(2, self.x_range-2), random.randint(2, self.y_range-2))
            for obs in self.obstacles:
                if not obs.intersects(Point(*point)):
                    return point
    
    def get_risky_point(self, point):
        danger_zone = set()
        for i in range(self.danger_dist * 2 + 1):
            x0 = point[0] + self.danger_dist - i
            for j in range(self.danger_dist * 2 + 1):
                x1 = point[1] + self.danger_dist - j
                danger_zone.add((x0, x1))
        return danger_zone

    def get_obs_bounds(self):
        obs_bounds = set()
        for obs in self.obstacles:
            obs_bounds.add(obs.bounds)

        return obs_bounds