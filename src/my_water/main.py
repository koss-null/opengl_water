from vispy import gloo, app, io
import numpy as np

from surface import *
from shaders import *
import time

def normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.sqrt(np.sum(vec * vec, axis=-1))[..., None]

class Canvas(app.Canvas):

    def __init__(self, surface, sky="fluffy_clouds.png", bed="seabed.png", shademap="shademap.png"):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface")
        gloo.set_state(clear_color=(0, 0, 0, 1), depth_test=True, blend=False)
        self.program = gloo.Program(VS, FS_triangle)
        self.program["a_position"] = surface.position()
        self.program["a_height"] = surface.get_heights_in_norm_coords()

        sun = np.array([0.7, 0.5, 0.7], dtype=np.float32)
        sun /= np.linalg.norm(sun)
        self.program["u_sun_direction"] = sun
        self.program["u_sun_color"] = np.array([0.7, 0.7, 0.6], dtype=np.float32)
        self.program["u_ambient_color"] = np.array([0.05, 0.5, 0.5], dtype=np.float32)

        self.sky = io.read_png(sky)
        self.program['u_sky_texture'] = gloo.Texture2D(self.sky, wrapping='repeat', interpolation='linear')
        self.bed = io.read_png(bed)
        self.program['u_bed_texture'] = gloo.Texture2D(self.bed, wrapping='repeat', interpolation='linear')
        self.shademap = io.read_png(shademap)
        self.program['u_shademap_texture'] = gloo.Texture2D(self.shademap, interpolation='nearest')

        self.eye_height = 2.5
        self.eye_position = [0, 0]
        self.program["u_eye_height"] = self.eye_height
        self.program["u_eye_position"] = self.eye_position
        self.program["u_alpha"] = 0.3
        self.program["u_bed_depth"] = 1

        self.triangles = gloo.IndexBuffer(surface.triangulation())

        self.camera = np.array([0, 0, 1])
        self.up = np.array([0, 1, 0])
        self.set_camera()

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
        self.program["a_height"] = surface.get_heights_in_norm_coords()
        self.program["a_normal"] = surface.normal()

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
            self.eye_position = [self.eye_position[0], self.eye_position[1] + 0.1]
            self.program["u_eye_position"] = self.eye_position
        elif event.key == 's':
            self.eye_position = [self.eye_position[0], self.eye_position[1] - 0.1]
            self.program["u_eye_position"] = self.eye_position
        elif event.key == 'a':
            self.eye_position = [self.eye_position[0] + 0.1, self.eye_position[1]]
            self.program["u_eye_position"] = self.eye_position
        elif event.key == 'd':
            self.eye_position = [self.eye_position[0] - 0.1, self.eye_position[1]]
            self.program["u_eye_position"] = self.eye_position
        elif event.key == '0':
            self.eye_position = [0, 0]
            self.program["u_eye_position"] = self.eye_position

    def set_camera(self):
        rotation = np.zeros((4, 4), dtype=np.float32)
        rotation[3, 3] = 1
        rotation[0, :3] = np.cross(self.up, self.camera)
        rotation[1, :3] = self.up
        rotation[2, :3] = self.camera
        world_view = rotation
        self.program['u_world_view'] = world_view.T
        self.program_point['u_world_view'] = world_view.T

    def rotate_camera(self, shift):
        right = np.cross(self.up, self.camera)
        new_camera = self.camera - right * shift[0] + self.up * shift[1]
        new_up = self.up - self.camera * shift[0]
        self.camera = normalize(new_camera)
        self.up = normalize(new_up)
        self.up = np.cross(self.camera, np.cross(self.up, self.camera))

    def screen_to_gl_coordinates(self, pos):
        return 2 * np.array(pos) / np.array(self.size) - 1

    def on_mouse_press(self, event):
        self.drag_start = self.screen_to_gl_coordinates(event.pos)

    def on_mouse_move(self, event):
        if not self.drag_start is None:
            pos = self.screen_to_gl_coordinates(event.pos)
            self.rotate_camera(pos - self.drag_start)
            self.drag_start = pos
            self.set_camera()
            self.update()

    def on_mouse_release(self, event):
        self.drag_start = None


if __name__ == '__main__':
    surface = NaturalWaves(size=(20, 20), max_height=0.9)
    surface.generate_random_waves(intensity=0)
    c = Canvas(surface)
    app.run()
