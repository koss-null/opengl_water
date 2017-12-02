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

    def wireframe(self):
        final_lines = []
        for row in range(0, self.size[1], 1):
            for dot in range(0, self.size[0], 1):
                if row + 1 == self.size[1] or dot + 1 == self.size[0]:
                    continue
                up_left = row * self.size[0] + dot
                up_right = up_left + 1
                dn_left = (row + 1) * self.size[0] + dot
                dn_right = dn_left + 1
                final_lines.append([up_left, up_right])
                final_lines.append([up_left, dn_left])

        return np.array(final_lines).astype(np.uint32)

    def triangulation(self):
        final_triangles = []
        for row in range(0, self.size[1], 1):
            for dot in range(0, self.size[0], 1):
                if row + 1 == self.size[1] or dot + 1 == self.size[0]:
                    continue
                up_left = row * self.size[0] + dot
                up_right = up_left + 1
                dn_left = (row + 1) * self.size[0] + dot
                dn_right = dn_left + 1
                final_triangles.append([up_left, up_right, dn_right])
                final_triangles.append([up_left, dn_left, dn_right])

        return np.array(final_triangles).astype(np.uint32)

    # def triangulation(self):
    #     a = np.indices((self.size[0] - 1, self.size[1] - 1))
    #     b = a + np.array([1, 0])[:, None, None]
    #     c = a + np.array([1, 1])[:, None, None]
    #     d = a + np.array([0, 1])[:, None, None]
    #
    #     a_r = a.reshape((2, -1))
    #     b_r = b.reshape((2, -1))
    #     c_r = c.reshape((2, -1))
    #     d_r = d.reshape((2, -1))
    #
    #     a_l = np.ravel_multi_index(a_r, self.size)
    #     b_l = np.ravel_multi_index(b_r, self.size)
    #     c_l = np.ravel_multi_index(c_r, self.size)
    #     d_l = np.ravel_multi_index(d_r, self.size)
    #
    #     abc = np.concatenate((a_l[..., None], b_l[..., None], c_l[..., None]), axis=-1)
    #     acd = np.concatenate((a_l[..., None], c_l[..., None], d_l[..., None]), axis=-1)
    #
    #     return np.concatenate((abc, acd), axis=0).astype(np.uint32)


    def one_random_wave(self):
        for i in range(0, 1):
            z_x_ind = random.randint(0, self.size[0] - 1)
            z_y_ind = random.randint(0, self.size[1] - 1)
            z_dot = float(random.randint(0, 100)) / 100.
            self.heights[z_x_ind][z_y_ind] = z_dot
            print("Generated wave of position " + str(z_dot) + " with coords " + str(z_x_ind) + " " + str(z_y_ind))


    @staticmethod
    def _force(height1, height2, dif_x, dif_y):
        difference = dif_x + dif_y
        coeff = 1
        connect_force = -0.005 * (difference**0.5) + 1
        sign = -1 if height2 > height1 else 1
        return sign * abs(height1 - height2) * connect_force * coeff

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

        n_nearest = 1
        # if self.heights[x][y] == 0 and self.speed[x][y] == 0:
        #     n_nearest = 1
        for i in range(max(x-n_nearest, 0), min(x+n_nearest, self.size[0]-1) + 1):
            for j in range(max(y-n_nearest, 0), min(y+n_nearest, self.size[1]-1) + 1):
                if i == x and j == y:
                    continue
                avg_vel += NaturalWaves._force(self.heights[x][y], self.heights[i][j], abs(x-i), abs(y-j))

        # if self.heights[x][y] == 0 and self.speed[x][y] == 0 and avg_vel != 0:
        #     n_nearest = 3
        #     for i in range(max(x - n_nearest, 0), min(x + n_nearest, self.size[0] - 1)):
        #         for j in range(max(y - n_nearest, 0), min(y + n_nearest, self.size[1] - 1)):
        #             if i == x and j == y:
        #                 continue
        #             avg_vel += NaturalWaves._force(self.heights[x][y], self.heights[i][j], abs(x - i), abs(y - j))

        return (avg_vel + g_force) * time

    def _count_height(self, z_next, x, y, time):
        speed_changing = self._count_avg_speed(x, y, -0)
        next_speed = self.speed_x[x][y] + speed_changing
        self.speed_x[x][y] = next_speed
        friction_coef = 0.925
        z_next[x][y] = self.heights[x][y] * friction_coef - next_speed * time

    def next_wave_mutation(self, time=0.005):
        # counting water mass center

        #z_next = np.zeros(self.size, dtype=np.float32)
        self.speed_x = self.speed
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self._count_height(self.heights, x, y, time)

        #self.heights = z_next
        self.speed = self.speed_x
        self._normalize()


class Surface():
    pass