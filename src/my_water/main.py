from vispy import gloo, app, io

from surface import *
import time

VS = ("""
#version 120

attribute vec2 a_position;
attribute float a_height;

void main (void) {
    gl_Position = vec4(a_position.xy, a_height, a_height);
}
""")

FS_triangle = ("""
""")

FS_point = """
#version 120

void main() {
    gl_FragColor = vec4(1, 1, 1, 1);

}
"""


class Canvas(app.Canvas):

    def __init__(self, surface):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface")
        gloo.set_state(clear_color=(0, 0, 0, 1), depth_test=False, blend=False)
        self.program = gloo.Program(VS, FS_point)
        self.program["a_position"] = surface.position()
        self.program["a_height"] = surface.heights

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
        self.program["a_height"] = surface.heights
        self.program.draw('points')

    def on_timer(self, event):
        self.t += 0.03
        self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def on_mouse_press(self, event):
        surface.one_random_wave()

if __name__ == '__main__':
    surface = NaturalWaves(size=(46, 46), max_height=0.1)
    surface.generate_random_waves(intensity=30)
    c = Canvas(surface)
    app.run()
