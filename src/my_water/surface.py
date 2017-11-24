import random

import numpy as np
from time import sleep
import threading

class NaturalWaves:

    def __init__(self, size=(100, 100), max_height=0.6):
        self.size = size
        self.max_height = max_height
        self.speed = np.zeros(self.size, dtype=np.float32)
        self.heights = np.ones(self.size, dtype=np.float32)

    def position(self):
        xy = np.empty(self.size + (2,), dtype=np.float32)
        xy[:, :, 0] = np.linspace(-1, 1, self.size[0])[:, None]
        xy[:, :, 1] = np.linspace(-1, 1, self.size[1])[None, :]
        return xy

    def generate_random_waves(self, intensity=4):
        z = np.ones(self.size, dtype=np.float32)

        for i in range(intensity):
            z_x_ind = random.randint(0, self.size[0] - 1)
            z_y_ind = random.randint(0, self.size[1] - 1)
            z_dot = 1 - float(random.randint(0, int(100 * (1 - self.max_height)))) / 100.
            z[z_x_ind][z_y_ind] = z_dot

        self.heights = np.array(z, dtype=np.float32)

        return z

    def one_random_wave(self):
        z_x_ind = random.randint(0, self.size[0] - 1)
        z_y_ind = random.randint(0, self.size[1] - 1)
        z_dot = 0.5 - float(random.randint(0, int(100 * (1 - self.max_height)))) / 100.
        self.heights[z_x_ind][z_y_ind] = z_dot


    @staticmethod
    def _force(height1, height2):
        connect_force = 0.8
        sign = -1 if height2 > height1 else 1
        return sign * abs(height1 - height2) * connect_force

    def _normalize(self):
        for i in range(0, len(self.heights)):
            for j in range(0, len(self.heights[i])):
                if self.heights[i][j] > 1:
                    self.heights[i][j] = 1
                elif self.heights[i][j] < self.max_height:
                    self.heights[i][j] = self.max_height

    # counts what speed will be 1 sec after
    def _count_avg_speed(self, x, y):
        avg_vel = 0.
        corpuscule_m = 1.
        g_force = -0.0006 * corpuscule_m
        time = 1.

        n_nearest = 3
        for i in range(max(x-n_nearest, 0), min(x+n_nearest, self.size[0]-1)):
            for j in range(max(y-n_nearest, 0), min(y+n_nearest, self.size[1]-1)):
                if i == x and j == y:
                    continue
                avg_vel += NaturalWaves._force(self.heights[x][y], self.heights[i][j])

        return (avg_vel + g_force) * time

    def next_wave_mutation(self, time=0.003):
        # counting water mass center

        z_next = np.empty(self.size, dtype=np.float32)
        for x in range(self.size[0]-1):
            for y in range(self.size[1]-1):
                def func():
                    speed_changing = self._count_avg_speed(x, y)
                    next_speed = self.speed[x][y] + speed_changing
                    self.speed[x][y] = next_speed
                    z_next[x][y] = self.heights[x][y] - next_speed * time

                t = threading.Thread(target=func())
                t.daemon = True
                t.start()

        self.heights = z_next
        self._normalize()


class Surface():
    pass