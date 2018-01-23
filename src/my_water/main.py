from vispy import gloo, app, io
import numpy as np

from surface import *
from shaders import *
import time

import matplotlib.pyplot as plt

def normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.sqrt(np.sum(vec * vec, axis=-1))[..., None]

class Canvas(app.Canvas):

    def _vec_norm(self, coords):
        abs = math.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2)
        if abs == 0:
            return np.array([0, 0, 0])
        return np.array([coords[0] / abs, coords[2] / abs, coords[2] / abs])

    def __init__(self, surface, sky="fluffy_clouds.png", bed="seabed.png", shademap_name="shademap.png"):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface")
        gloo.set_state(clear_color=(0, 0, 0, 1), depth_test=True, blend=True)
        self.program = gloo.Program(VS, FS_triangle)
        self.program["a_position"] = surface.position()
        self.program["u_surf_size"] = self.size

        height_texture = surface.get_heights_in_norm_coords()
        self.program["u_height"] = gloo.Texture2D(height_texture, wrapping='repeat', interpolation='linear')

        depth_texture = surface.get_bed_depth()
        self.program["u_bed_depth"] = gloo.Texture2D(depth_texture, wrapping='repeat', interpolation='linear')

        self.program["a_dot"] = surface.dot_types

        sun = np.array([0., 0.5, 1], dtype=np.float32)
        self.sun = sun / np.linalg.norm(sun)
        self.program["u_sun_direction"] = self.sun
        self.program["u_sun_color"] = np.array([1, 0.8, 0.6], dtype=np.float32)
        ambient = [0.4, 0.4, 0.4]
        self.program["u_ambient_color"] = np.array(ambient, dtype=np.float32)

        self.sky = io.read_png(sky)
        self.program['u_sky_texture'] = gloo.Texture2D(self.sky, wrapping='mirrored_repeat', interpolation='linear')
        self.bed = io.read_png(bed)
        self.program['u_bed_texture'] = gloo.Texture2D(self.bed, wrapping='repeat', interpolation='linear')

        self.eye_height = 4.5
        self.eye_position = np.array([0., 0.])
        self.program["u_eye_height"] = self.eye_height
        self.program["u_eye_position"] = self.eye_position
        self.program["u_alpha"] = 0.4

        self.angle_x, self.angle_y, self.angle_z = 0, 0, 0
        self.program["u_angle"] = np.array([self.angle_x, self.angle_y, self.angle_z])

        self.show_bed = 0
        self.program["u_show_bed"] = self.show_bed
        self.show_sky = 0
        self.program["u_show_sky"] = self.show_sky

        self.program["test"] = 0.

        self.triangles = gloo.IndexBuffer(surface.triangulation())
        # self.normal = surface.normal()
        # self.program["u_normal"] = gloo.Texture2D(self.normal, interpolation='linear')

        self.camera = np.array([0, 0, 1])
        self.up = np.array([0, 1, 0])

        self.t = 0
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.activate_zoom()
        self.show()

    def activate_zoom(self):
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_draw(self, event):
        gloo.clear()
        surface.next_wave_mutation()
        height_texture = surface.get_heights_in_norm_coords()
        self.program["u_height"] = gloo.Texture2D(height_texture,  wrapping='repeat', interpolation='linear')
        # plt.imshow(surface.get_heights_in_norm_coords()) #Needs to be in row,col orders
        # plt.savefig('figure1.png')

        self.program.draw('triangles', self.triangles)

    def on_timer(self, event):
        self.t += 1
        self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def on_key_press(self, event):
        if event.key == 'Escape':
            self.close()
        elif event.key == ' ':
            surface.one_random_wave()
        elif event.key == '-':
            self.eye_height += 0.1
            self.program["u_eye_height"] = self.eye_height
        elif event.key == '=':
            self.eye_height -= 0.1
            self.program["u_eye_height"] = self.eye_height
        elif event.key == 'w':
            self.angle_x += 0.1
            self.program["u_angle"] = np.array([self.angle_x, self.angle_y, self.angle_z])
        elif event.key == 's':
            self.angle_x -= 0.1
            self.program["u_angle"] = np.array([self.angle_x, self.angle_y, self.angle_z])
        elif event.key == 'a':
            self.angle_y -= 0.1
            self.program["u_angle"] = np.array([self.angle_x, self.angle_y, self.angle_z])
        elif event.key == 'd':
            self.angle_y += 0.1
            self.program["u_angle"] = np.array([self.angle_x, self.angle_y, self.angle_z])
        elif event.key == 'z':
            self.angle_z += 0.1
            self.program["u_angle"] = np.array([self.angle_x, self.angle_y, self.angle_z])
        elif event.key == 'x':
            self.angle_z -= 0.1
            self.program["u_angle"] = np.array([self.angle_x, self.angle_y, self.angle_z])
        elif event.key == '0':
            self.eye_height = 3
            self.program["u_eye_height"] = self.eye_height
            self.angle_x, self.angle_y, self.angle_z = 0., 0., 0.
            self.program["u_angle"] = np.array([self.angle_x, self.angle_y, self.angle_z])
        elif event.key == 'b':
            self.show_bed = (self.show_bed + 1) % 2
            self.program["u_show_bed"] = self.show_bed
        elif event.key == 'k':
            self.show_sky = (self.show_sky + 1) % 2
            self.program["u_show_sky"] = self.show_sky
        elif event.key == 't':
            self.program["test"] = 1.
        elif event.key == 'r':
            self.program["test"] = 0.


    def screen_to_gl_coordinates(self, pos):
        return 2 * np.array(pos) / np.array(self.size) - 1

    def on_mouse_press(self, event):
        self.drag_start = self.screen_to_gl_coordinates(event.pos)

    def on_mouse_move(self, event):
        lol = 1
        # if not self.drag_start is None:
        #     pos = self.screen_to_gl_coordinates(event.pos)
        #     self.rotate_camera(pos - self.drag_start)
        #     self.drag_start = pos
        #     self.set_camera()
        #     self.update()

    def on_mouse_release(self, event):
        self.drag_start = None


if __name__ == '__main__':
    surface = NaturalWaves(size=(25, 25), max_height=0.9)
    # surface = RungeWaves(size=(30, 30), max_height=0.9)
    # surface = GeomethricFigure(size=(50, 50), max_height=1)
    surface.generate_random_waves(intensity=100)
    c = Canvas(surface)
    app.run()
