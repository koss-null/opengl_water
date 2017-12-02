from vispy import gloo, app, io

from surface import *
from shaders import *
import time


class Canvas(app.Canvas):

    def __init__(self, surface):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface")
        gloo.set_state(clear_color=(0, 0, 0, 1), depth_test=False, blend=True)
        self.program = gloo.Program(VS, FS_triangle)
        self.program["a_position"] = surface.position()
        self.program["a_height"] = surface.get_heights_in_norm_coords()

        self.triangles = gloo.IndexBuffer(surface.wireframe())

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

        #self.program.draw('points')
        self.program.draw('lines', self.triangles)

    def on_timer(self, event):
        self.t += 0.7
        self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def on_mouse_press(self, event):
        surface.one_random_wave()

if __name__ == '__main__':
    surface = NaturalWaves(size=(15, 15), max_height=0.6)
    surface.generate_random_waves(intensity=10)
    c = Canvas(surface)
    app.run()
