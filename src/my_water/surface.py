import random

import numpy as np
from time import sleep
import multiprocessing
import math


class GeomethricFigure:

    def __init__(self, size=(10, 10), max_height=0.6):
        self.size = size
        self.grad = None
        self.max_height = max_height
        self.heights = np.zeros(self.size, dtype=np.float32)
        self.dot_types = np.zeros(self.size[0] * self.size[1], dtype=np.float32)

    def position(self):
        xy = np.empty(self.size + (2,), dtype=np.float32)
        xy[:, :, 0] = np.linspace(-1, 1, self.size[0])[:, None]
        xy[:, :, 1] = np.linspace(-1, 1, self.size[1])[None, :]
        return xy

    def generate_random_waves(self, intensity=4):
        z = np.zeros(self.size, dtype=np.float32)

        for y in range(0, self.size[0]):
            for x in range(0, self.size[1]):
                z[x][y] = (x-25)**4 + (y-25)**3 - y*7

        self.heights = np.array(z, dtype=np.float32)

        self._normalize()
        self.heights = self.get_heights_in_norm_coords()
        return self.heights

    def one_random_wave(self):
        print("mock")

    # turns -1;1 coords into 0;1
    def get_heights_in_norm_coords(self):
        z_norm = np.array(self.heights, dtype=np.float32)
        i = 0
        for z in self.heights:
            zz = 1 - (1 + z) * 0.5
            z_norm[i] = zz
            i += 1
        return z_norm

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

    def _normalize(self):
        max_val = -10000.
        for i in range(0, len(self.heights)):
            for j in range(0, len(self.heights[i])):
                if self.heights[i][j] > max_val:
                    max_val = self.heights[i][j]

        for i in range(0, len(self.heights)):
            for j in range(0, len(self.heights[i])):
                self.heights[i][j] /= abs(max_val)

    def _vec_norm(self, coords):
        abs = math.sqrt(coords[0]**2 + coords[1] **2 + coords[2]**2)
        if abs == 0:
            return np.array([0, 0, 0])
        return np.array([coords[0] / abs, coords[1] / abs, coords[2] / abs])

    def normal(self):
        if self.grad != None:
            return self.grad

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
                x2, y2, z2 = 0., 0., heights[int(triangle[1] / self.size[1])][triangle[1] % self.size[1]]
                x3, y3, z3 = 1., 0., heights[int(triangle[2] / self.size[1])][triangle[2] % self.size[1]]

            # coeffs of plane equation
            A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
            B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
            C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            D = -(x1 * (y2*z3 - y3*z2) + x2 * (y3*z1 - y1*z3) + x3 * (y1*z2 - y2*z1))

            grad[triangle[0]] = self._vec_norm([A, B, C])
            grad[triangle[1]] = self._vec_norm([A, B, C])
            grad[triangle[2]] = self._vec_norm([A, B, C])

            type = (type + 1) % 2    # changing type into next one

        self.grad = grad
        return grad

    def get_bed_depth(self):
        z = np.zeros(self.size, dtype=np.float32)

        for y in range(0, self.size[0]):
            for x in range(0, self.size[1]):
                z[x][y] = -1

        return z

    def next_wave_mutation(self, time=0.0005):
        self._normalize()


class NaturalWaves():
    def __init__(self, size=(10, 10), max_height=0.6):
        self.size = size
        self.max_height = max_height
        self.speed = np.zeros(self.size, dtype=np.float32)
        self.heights = np.zeros(self.size, dtype=np.float32)
        self.normals = np.zeros(self.size, dtype=np.float32)
        self.dot_types = np.zeros(self.size[0] * self.size[1], dtype=np.float32)

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

    def one_random_wave(self):
        for i in range(0, 1):
            z_x_ind = self.size[0] / 2 #random.randint(0, self.size[0] - 1)
            z_y_ind = self.size[1] / 2 #random.randint(0, self.size[1] - 1)
            x_ind_start, x_ind_end = z_x_ind - 5, z_x_ind + 5
            y_ind_start, y_ind_end = z_y_ind - 5, z_y_ind + 5
            z_dot = float(random.randint(0, 100)) / 100.
            for x in range(x_ind_start, x_ind_end):
                for y in range(y_ind_start, y_ind_end):
                    difference = abs(x - z_x_ind) + abs(y - z_y_ind)
                    connect_force = -0.5 * (difference ** 0.5) + 1.5
                    self.heights[x][y] = z_dot * connect_force
            print("Generated wave of position " + str(z_dot) + " with coords " + str(z_x_ind) + " " + str(z_y_ind))

    # turns -1;1 coords into 0;100
    def get_heights_in_norm_coords(self):
        z_norm = [[]] * len(self.heights)
        i = 0
        for z in self.heights:
            zz = (1 - (1 + z) * 0.5) * 100
            rgb = []
            j = 0
            for item in zz:
                if (j % 2 == 0):
                    self.dot_types[i * len(self.heights) + j] = 1
                    rgb.append([int(item), int(zz[min(j + 1, len(zz)-1)]), int(self.heights[min(i + 1, len(self.heights)-1)][j])])
                else:
                    self.dot_types[i * len(self.heights) + j] = 2
                    rgb.append([int(zz[max(j - 1, 0)]), int(item), int(self.heights[min(i + 1, len(self.heights)-1)][max(j-1, 0)])])
                j += 1
            z_norm[i] = np.array(rgb, dtype=np.uint8)
            i += 1
        return np.array(z_norm, dtype=np.uint8)

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
        return np.array([coords[0] / abs, coords[1] / abs, coords[2] / abs])

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
                x2, y2, z2 = 0., 0., heights[int(triangle[1] / self.size[1])][triangle[1] % self.size[1]]
                x3, y3, z3 = 1., 0., heights[int(triangle[2] / self.size[1])][triangle[2] % self.size[1]]

            # coeffs of plane equation
            A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
            B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
            C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            D = -(x1 * (y2*z3 - y3*z2) + x2 * (y3*z1 - y1*z3) + x3 * (y1*z2 - y2*z1))

            nrml = np.array([int(A * 10000), int(B * 10000), int(C * 10)], dtype=np.int16)
            grad[triangle[0]] = nrml
            grad[triangle[1]] = nrml
            grad[triangle[2]] = nrml

            type = (type + 1) % 2    # changing type into next one

        self.normals = grad
        return grad

    @staticmethod
    def _force(height1, height2, dif_x, dif_y):
        difference = dif_x + dif_y
        coeff = 200
        connect_force = -0.5 * (difference**0.5) + 1.5
        sign = -1 if height2 > height1 else 1
        return sign * abs(height1 - height2) * connect_force * coeff

    # counts what speed will be 1 sec after
    def _count_avg_speed(self, x, y, lower_bound):
        avg_vel = 0.
        corpuscule_m = 3.
        g_force = 0.0009 * corpuscule_m if self.heights[x][y] < lower_bound else -0.00009 * self.heights[x][y]
        time = 1.

        n_nearest = 2
        # if self.heights[x][y] == 0 and self.speed[x][y] == 0:
        #     n_nearest = 1
        for i in range(max(x-n_nearest, 0), min(x+n_nearest, self.size[0]-1) + 1):
            for j in range(max(y-n_nearest, 0), min(y+n_nearest, self.size[1]-1) + 1):
                if i == x and j == y:
                    continue
                avg_vel += self._force(self.heights[x][y], self.heights[i][j], abs(x-i), abs(y-j))

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
        friction_coef = 0.9925
        next_speed = self.speed[x][y] * friction_coef + speed_changing
        self.speed_x[x][y] = next_speed
        z_next[x][y] = self.heights[x][y] - next_speed * time

    def next_wave_mutation(self, time=0.0005):
        # counting water mass center

        z_next = np.zeros(self.size, dtype=np.float32)
        self.speed_x = self.speed
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self._count_height(z_next, x, y, time)

        self.heights = z_next
        self.speed = self.speed_x
        self._normalize()

    def get_bed_depth(self):
        z = np.zeros(self.size, dtype=np.float32)

        for y in range(0, self.size[0]):
            for x in range(0, self.size[1]):
                z[x][y] = min((x - self.size[1]/2) ** 2, 15)

        #normalization
        max_val = -10000.
        for i in range(0, len(z)):
            for j in range(0, len(z[i])):
                if z[i][j] > max_val:
                    max_val = z[i][j]

        for i in range(0, len(z)):
            for j in range(0, len(z[i])):
                z[i][j] /= abs(max_val)

        # getting GLSL coords
        z_norm = z
        i = 0
        for j in z:
            zz = 1 - (1 + j) * 0.5
            z_norm[i] = zz
            i += 1
        return z_norm


class RungeWaves():
    def __init__(self, size=(10, 10), max_height=0.6):
        self.size = size
        self.max_height = max_height
        self.heights = np.zeros(self.size, dtype=np.float32)
        self.speed = np.zeros(self.size, dtype=np.float32)
        self.speed_last = np.zeros(self.size, dtype=np.float32)

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

    def one_random_wave(self):
        for i in range(0, 1):
            z_x_ind = random.randint(0, self.size[0] - 1) #self.size[0] / 2
            z_y_ind = random.randint(0, self.size[1] - 1) #self.size[1] / 2
            x_ind_start, x_ind_end = z_x_ind - 0, z_x_ind + 1
            y_ind_start, y_ind_end = z_y_ind - 0, z_y_ind + 1
            z_dot = float(random.randint(0, 100)) / 100.
            for x in range(x_ind_start, x_ind_end):
                for y in range(y_ind_start, y_ind_end):
                    difference = abs(x - z_x_ind) + abs(y - z_y_ind)
                    connect_force = -0.5 * (difference ** 0.5) + 1.5
                    self.heights[x][y] = z_dot * connect_force
            print("Generated wave of position " + str(z_dot) + " with coords " + str(z_x_ind) + " " + str(z_y_ind))

    # turns -1;1 coords into 0;1
    def get_heights_in_norm_coords(self):
        z_norm = np.array(self.heights, dtype=np.float32)
        i = 0
        for z in self.heights:
            zz = 1 - (1 + z) * 0.5
            z_norm[i] = zz
            i += 1
        return z_norm

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

    def _normalize(self):
        for i in range(0, len(self.heights)):
            for j in range(0, len(self.heights[i])):
                if self.heights[i][j] < -1:
                    self.heights[i][j] = -1
                elif self.heights[i][j] > self.max_height:
                    self.heights[i][j] = self.max_height

    def _vec_norm(self, coords):
        abs = math.sqrt(coords[0]**2 + coords[1] **2 + coords[2]**2)
        if abs == 0:
            return np.array([0, 0, 0])
        return np.array([coords[0] / abs, coords[1] / abs, coords[2] / abs])

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
                x2, y2, z2 = 0., 0., heights[int(triangle[1] / self.size[1])][triangle[1] % self.size[1]]
                x3, y3, z3 = 1., 0., heights[int(triangle[2] / self.size[1])][triangle[2] % self.size[1]]

            # coeffs of plane equation
            A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
            B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
            C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            D = -(x1 * (y2*z3 - y3*z2) + x2 * (y3*z1 - y1*z3) + x3 * (y1*z2 - y2*z1))

            grad[triangle[0]] = self._vec_norm([A, B, C])
            grad[triangle[1]] = self._vec_norm([A, B, C])
            grad[triangle[2]] = self._vec_norm([A, B, C])

            type = (type + 1) % 2    # changing type into next one

        return grad

    #####################################
    #####################################
    #####################################
    #####################################

    def _count_height_dummy(self, z_next, x, y, time):
        xp = self.heights[x+1][y] if x+1 < self.size[0] else 0
        xm = self.heights[x-1][y] if x > 0 else 0
        yp = self.heights[x][y+1] if y+1 < self.size[1] else 0
        ym = self.heights[x][y-1] if y > 0 else 0
        a = self.speed[x][y] ** 2 + (self.size[0]**2) * (xp + xm + yp + ym - 4 * self.heights[x][y])

        self.speed[x][y] += a * time
        z_next[x][y] = self.heights[x][y] + self.speed[x][y] * time + (a * time**2) / 2

    def _runge_func(self, x, y, time):
        v_cur = self.speed[int(x)][int(y)]
        v_last = self.speed_last[int(x)][int(y)]

        if int(x) != x:
            v_cur = (self.speed[int(x)][y] + self.speed[int(x) + 1][y]) / 2
            v_last = (self.speed_last[int(x)][y] + self.speed_last[int(x) + 1][y]) / 2
        elif int(y) != y:
            v_cur = (self.speed[x][int(y)] + self.speed[x][int(y) + 1]) / 2
            v_last = (self.speed_last[x][int(y)] + self.speed_last[x][int(y) + 1]) / 2
        return float(v_cur - v_last)/time

    def _count_height_runge(self, z_next, x, y, time):
        step = float(1/self.size[0])

        kx1 = self.speed[x][y] ** 2 * self._runge_func(x, y, time)
        kx2 = self.speed[x][y] ** 2 * self._runge_func(float(x) + step/2, float(y) + step/2 * kx1, time)
        kx3 = self.speed[x][y] ** 2 * self._runge_func(float(x) + step/2, float(y) + step/2 * kx2, time)
        kx4 = self.speed[x][y] ** 2 * self._runge_func(float(x) + step, float(y) + step * kx3, time)
        Lx = self.speed[x][y] + 6/self.size[0] * (kx1 + 2*kx2 + 2*kx3 + kx4)

        ky1 = self.speed[x][y] ** 2 * self._runge_func(x, y, time)
        ky2 = self.speed[x][y] ** 2 * self._runge_func(float(x) + step / 2, float(y) + step / 2 * ky1, time)
        ky3 = self.speed[x][y] ** 2 * self._runge_func(float(x) + step / 2, float(y) + step / 2 * ky2, time)
        ky4 = self.speed[x][y] ** 2 * self._runge_func(float(x) + step, float(y) + step * ky3, time)
        Ly = self.speed[x][y] + 6 / self.size[0] * (ky1 + 2 * ky2 + 2 * ky3 + ky4)

        dv = Lx + Ly
        self.speed_last[x][y] = self.speed[x][y]
        self.speed[x][y] += dv
        z_next[x][y] += self.heights[x][y] + self.speed[x][y] * time


    def next_wave_mutation(self, time=0.009):
        # counting water mass center

        z_next = np.zeros(self.size, dtype=np.float32)
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self._count_height_dummy(z_next, x, y, time)
                # self._count_height_runge(z_next, x, y, time)

        self.heights = z_next
        self._normalize()

    def get_bed_depth(self):
        z = np.zeros(self.size, dtype=np.float32)

        for y in range(0, self.size[0]):
            for x in range(0, self.size[1]):
                z[x][y] = (x - self.size[1]/2) ** 2

        #normalization
        max_val = -10000.
        for i in range(0, len(z)):
            for j in range(0, len(z[i])):
                if z[i][j] > max_val:
                    max_val = z[i][j]

        for i in range(0, len(z)):
            for j in range(0, len(z[i])):
                z[i][j] /= abs(max_val)

        # getting GLSL coords
        z_norm = z
        i = 0
        for j in z:
            zz = 1 - (1 + j) * 0.5
            z_norm[i] = zz
            i += 1
        return z_norm


class Surface():
    pass