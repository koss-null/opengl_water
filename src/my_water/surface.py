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

        self.triangulation = final_triangles
        return np.array(final_triangles).astype(np.uint32)

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
        coeff = 100
        connect_force = -0.5 * (difference**0.5) + 1.5
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

    def _vec_norm(self, coords):
        abs = math.sqrt(coords[0]**2 + coords[1] **2 + coords[2]**2)
        if abs == 0:
            return np.array([0, 0, 0])
        return np.array([coords[0] / abs, coords[2] / abs, coords[2] / abs])

    def normal(self):
        grad = [np.array([0., 0., 0.])] * (self.size[0]*self.size[1])
        type = 0 # type of triangle
        heights = self.get_heights_in_norm_coords()
        for triangle in self.triangulation:
            if type == 0:
                x1, y1, z1 = 0., 1., heights[int(triangle[0] / self.size[1])][triangle[0] % self.size[1]]
                x2, y2, z2 = 1., 1., heights[int(triangle[1] / self.size[1])][triangle[1] % self.size[1]]
                x3, y3, z3 = 1., 0., heights[int(triangle[2] / self.size[1])][triangle[2] % self.size[1]]
            else:
                x1, y1, z1 = 0., 1., heights[int(triangle[0] / self.size[1])][triangle[0] % self.size[1]]
                x2, y2, z2 = 1., 0., heights[int(triangle[1] / self.size[1])][triangle[1] % self.size[1]]
                x3, y3, z3 = 0., 0., heights[int(triangle[2] / self.size[1])][triangle[2] % self.size[1]]

            # coeffs of plane equation
            A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
            B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
            C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            D = -(x1 * (y2*z3 - y3*z2) + x2 * (y3*z1 - y1*z3) + x3 * (y1*z2 - y2*z1))

            grad[triangle[0]] = np.add(grad[triangle[0]], self._vec_norm([A, B, C]))
            grad[triangle[1]] = np.add(grad[triangle[1]], self._vec_norm([A, B, C]))
            grad[triangle[2]] = np.add(grad[triangle[2]], self._vec_norm([A, B, C]))

            type = (type + 1) % 2    # changing type into next one

        return grad

    # counts what speed will be 1 sec after
    def _count_avg_speed(self, x, y, lower_bound):
        avg_vel = 0.
        corpuscule_m = 1.
        g_force = 0.0009 * corpuscule_m if self.heights[x][y] < lower_bound else -0.00009 * self.heights[x][y]
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
        next_speed = self.speed[x][y] + speed_changing
        self.speed_x[x][y] = next_speed
        friction_coef = 0.9925
        z_next[x][y] = self.heights[x][y] * friction_coef - next_speed * time

    def next_wave_mutation(self, time=0.0001):
        # counting water mass center

        z_next = np.zeros(self.size, dtype=np.float32)
        self.speed_x = self.speed
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self._count_height(z_next, x, y, time)

        self.heights = z_next
        self.speed = self.speed_x
        self._normalize()


class Surface():
    pass