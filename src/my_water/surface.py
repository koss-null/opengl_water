import random

import numpy as np
from time import sleep
import multiprocessing
import math


class NaturalWaves:

    def __init__(self, size=(10, 10), max_height=0.6):
        self.size = size
        self.max_height = max_height
        self.speed = np.zeros(self.size, dtype=np.float32)
        self.heights = np.zeros(self.size, dtype=np.float32)
        self.pool = multiprocessing.Pool(processes=4)

    def position(self):
        xy = np.empty(self.size + (2,), dtype=np.float32)
        xy[:, :, 0] = np.linspace(-1, 1, self.size[0])[:, None]
        xy[:, :, 1] = np.linspace(-1, 1, self.size[1])[None, :]
        return xy

    def generate_random_waves(self, intensity=4):
        z = np.zeros(self.size, dtype=np.float32)

        for i in range(intensity):
            z_x_ind = random.randint(0, self.size[0] - 1)
            z_y_ind = random.randint(0, self.size[1] - 1)
            z_dot = float(random.randint(0, int(100 * (1 - self.max_height)))) / 100.
            z[z_x_ind][z_y_ind] = z_dot

        self.heights = np.array(z, dtype=np.float32)

        return z

    # turns -1;1 coords into 0;1
    def get_heights_in_norm_coords(self):
        z_norm = []
        for z in self.heights:
            zz = 1 - (1 + z) * 0.5
            z_norm.append(zz)
        return np.array(z_norm, dtype=np.float32)

    def one_random_wave(self):
        for i in range(0, 3):
            z_x_ind = random.randint(0, self.size[0] - 1)
            z_y_ind = random.randint(0, self.size[1] - 1)
            z_dot = float(random.randint(0, 100)) / 100.
            self.heights[z_x_ind][z_y_ind] = z_dot
            print("Generated wave of position " + str(z_dot) + " with coords " + str(z_x_ind) + " " + str(z_y_ind))


    @staticmethod
    def _force(height1, height2, dif_x, dif_y):
        difference = dif_x + dif_y
        connect_force = -0.01 * (difference**0.5) + 1
        sign = -1 if height2 > height1 else 1
        return sign * abs(height1 - height2) * connect_force

    def _normalize(self):
        for i in range(0, len(self.heights)):
            for j in range(0, len(self.heights[i])):
                if self.heights[i][j] < -1:
                    self.heights[i][j] = -1
                    self.speed[i][j] = 0.
                elif self.heights[i][j] > self.max_height:
                    self.heights[i][j] = self.max_height
                    self.speed[i][j] = 0.

    # counts what speed will be 1 sec after
    def _count_avg_speed(self, x, y, lower_bound):
        avg_vel = 0.
        corpuscule_m = 1.
        g_force = 0.0009 * corpuscule_m if self.heights[x][y] < lower_bound else -0.0009 * self.heights[x][y]
        time = 1.

        n_nearest = 2
        for i in range(max(x-n_nearest, 0), min(x+n_nearest, self.size[0]-1)):
            for j in range(max(y-n_nearest, 0), min(y+n_nearest, self.size[1]-1)):
                if i == x and j == y:
                    continue
                avg_vel += NaturalWaves._force(self.heights[x][y], self.heights[i][j], abs(x-i), abs(y-j))

        return (avg_vel + g_force) * time

    def _count_height(self, z_next, x, y, time):
        speed_changing = self._count_avg_speed(x, y, -0)
        next_speed = self.speed[x][y] + speed_changing
        self.speed[x][y] = next_speed
        friction_coef = 0.925
        z_next[x][y] = self.heights[x][y] * friction_coef - next_speed * time

    def next_wave_mutation(self, time=0.005):
        # counting water mass center

        z_next = np.zeros(self.size, dtype=np.float32)
        jobs = []
        times = 0
        for x in range(self.size[0]-1):
            for y in range(self.size[1]-1):
                self._count_height(z_next, x, y, time)

        self.heights = z_next
        self._normalize()


class Surface():
    pass