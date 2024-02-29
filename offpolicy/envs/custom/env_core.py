import numpy as np
from shapely.geometry import Polygon, Point
from shapely import intersects

import imageio

import os
import sys

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from .env_2d import map, plotting, Astar  # noqa: E402
from .env_2d.car_racing import CarRacing
# from utils.util import timethis

ACT_SPACE = [[0, 0, 0], [-0.6, 0, 0], [0.6, 0, 0], [0, 0.2, 0], [0, 0, 0.8]]


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        self.agent_num = 4  # number of agent
        self.obs_dim = 12  # observation dimension of agents
        self.action_dim = 5  # set the action dimension of agents
        self.guide_point_num = 100  # number of guide point
        self.map = map.Map()  # 2d env map
        self.width = self.map.x_range
        self.height = self.map.y_range
        self.car_env = CarRacing()

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """

        # 随机的 agent 位置
        self.car_center = np.array(self.map.random_point()).astype(float)
        # 目标位置
        self.dest = np.array(self.map.random_point())
        self.last_position = self.car_center
        # reset car env
        self.car_env.reset(car_pos=self.car_center)

        # guide point
        self.guide_points = self.get_guide_point(step=4)
        self.nearest_point = self.next_guide_point()

        # 智能体观测集合
        sub_agent_obs = self.get_sub_agent_obs(self.nearest_point)

        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        # input actions is one_hot
        actions = self._one_hot_to_actions(actions)
        self.car_env.step(actions)
        self.car_center = np.array(self.car_env.car.hull.position)

        # get next guide point
        self.nearest_point = self.next_guide_point()

        # observations after actions
        sub_agent_obs = self.get_sub_agent_obs(self.nearest_point)

        # information of each agent
        sub_agent_info = [{} for _ in range(self.agent_num)]

        sub_agent_reward = []
        sub_agent_done = []
        wheels_pos = ((w.position.x, w.position.y) for w in self.car_env.car.wheels)
        car = Polygon(wheels_pos)

        # Check termination conditions
        if self.is_target(car):
            sub_agent_done = [[True] for _ in range(self.agent_num)]
            sub_agent_reward = [[np.array(1000)] for _ in range(self.agent_num)]
            self.agents = []
        elif self.map.is_collision(car):
            sub_agent_done = [[True] for _ in range(self.agent_num)]
            sub_agent_reward = [[np.array(-100)] for _ in range(self.agent_num)]
            self.agents = []
        else:
            sub_agent_done = [[False] for _ in range(self.agent_num)]
            sub_agent_reward = self.get_reward(car)

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def is_target(self, car):
        return intersects(car, Point(self.dest[0], self.dest[1]))

    def get_reward(self, car):
        reward = 0

        # movement reward
        last_dist = np.linalg.norm(self.nearest_point - self.last_position)
        cur_dist = np.linalg.norm(self.nearest_point - self.car_center)
        diff = last_dist - cur_dist
        reward += diff

        # guide point reward
        if car.intersects(Point(self.nearest_point)):
            reward += 10
            # remove the nearest point
            self.guide_points = self.guide_points[1:]

        # update variables
        self.last_position = self.car_center

        return [[reward if reward > 0 else -2.0] for _ in range(self.agent_num)]

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            plot = plotting.Plotting(target=self.dest)
            plot.plot_map()
            wheels = [(w.position.x, w.position.y) for w in self.car_env.car.wheels]
            plot.plot_car(wheels)
            plot.plot_guide_point(self.guide_points)
            image = plot.save_image()

            return image

    def get_guide_point(self, step=1):
        start = tuple(self.car_center.astype(int).tolist())
        end = tuple(self.dest.astype(int).tolist())
        astar = Astar.AStar(start, end, "euclidean")
        path, _ = astar.searching()

        # return guide points
        path.reverse()
        return np.array(path[step:-1:step])

    def next_guide_point(self):
        while len(self.guide_points) > 1:
            first_point = self.guide_points[0]
            second_point = self.guide_points[1]
            angle = (second_point - first_point).dot(self.car_center - first_point)

            # 夹角是钝角
            if angle < 0:
                return first_point
            else:
                self.guide_points = self.guide_points[1:]
        return self.dest

    def get_sub_agent_obs(self, nearest_point):
        sub_agent_obs = []
        for i in range(self.agent_num):
            w = self.car_env.car.wheels[i]
            w_position = np.array([w.position.x, w.position.y])

            sub_obs = np.reshape(
                [
                    w_position,
                    np.array([w.omega, w.phase]),
                    self.car_center - w_position,
                    nearest_point - w_position,
                    self.car_center,  # global state
                    nearest_point,  # global state
                ],
                self.obs_dim,
            )

            sub_agent_obs.append(sub_obs)

        return sub_agent_obs

    def _one_hot_to_actions(self, one_hot_actions):
        action_index = np.argmax(one_hot_actions, axis=1)
        actions = [ACT_SPACE[action_index[i]] for i in range(self.agent_num)]
        return np.array(actions)


# @timethis
def env_test(times=10, render=False, mode="rgb_array"):
    """
    test the validation of env
    """
    env = EnvCore()

    for i in range(times):
        env.reset()

        all_frames = []
        if render:
            image = env.render(mode=mode)
            all_frames.append(image)

        step = 0
        episode_reward = 0
        for _ in range(1000):
            # actions = np.random.random(size=(env.agent_num,)) * 2 - 1
            # actions = np.expand_dims(env.dest - env.car_center, 0).repeat(env.agent_num, 0) / 10
            action_space = env.car_env.action_space
            # actions = np.array([[0.5, 1., 0.], [-0.5, 1., 0.], [0., 0., 0.], [0., 0., 0.]])
            actions = np.array([action_space.sample() for i in range(env.agent_num)])
            result = env.step(actions=actions)
            if render:
                all_frames.append(env.render())
            step += 1

            reward, done = result[1], result[2]
            episode_reward += reward[0][0]
            if np.all(done):
                break

        if render and mode == "rgb_array":
            import os
            import time

            image_dir = os.path.dirname(__file__) + "/" + "image"
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            time_now = int(time.time() * 1000)
            gif_save_path = image_dir + f"/{time_now}_{step}_{episode_reward:.2f}.gif"
            imageio.mimsave(gif_save_path, all_frames, duration=1, loop=0)


if __name__ == "__main__":
    env_test(times=5, render=True)
