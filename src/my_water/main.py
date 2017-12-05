from vispy import gloo, app, io
import numpy as np

from surface import *
from shaders import *
import time


class Canvas(app.Canvas):

    def __init__(self, surface):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface")
        gloo.set_state(clear_color=(0, 0, 0, 1), depth_test=True, blend=False)
        self.program = gloo.Program(VS, FS_triangle)
        self.program["a_position"] = surface.position()
        self.program["a_height"] = surface.get_heights_in_norm_coords()

        sun = np.array([1, 0, 1], dtype=np.float32)
        sun /= np.linalg.norm(sun)
        self.program["u_sun_direction"] = sun
        self.program["u_sun_color"] = np.array([0.7, 0.7, 0], dtype=np.float32)
        self.program["u_ambient_color"] = np.array([0.1, 0.0, 0.5], dtype=np.float32)

        self.triangles = gloo.IndexBuffer(surface.triangulation())

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

        self.program.draw('lines', self.triangles)

    def on_timer(self, event):
        self.t += 0.7
        self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def on_mouse_press(self, event):
        surface.one_random_wave()

if __name__ == '__main__':
    surface = NaturalWaves(size=(25, 25), max_height=0.99)
    surface.generate_random_waves(intensity=10)
    c = Canvas(surface)
    app.run()
