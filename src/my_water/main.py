from vispy import gloo, app, io

from surface import *
import time

VS = ("""
#version 120

attribute vec2 a_position;
attribute float a_height;

varying float v_z;

void main (void) {
    v_z = a_height;
    gl_Position = vec4(a_position.xy, a_height, 1);
}
""")

FS_triangle = ("""
#version 120

varying float v_z;

void main() {
    vec3 rgb=mix(vec3(1,0.5,0),vec3(0,0.5,1.0),v_z);
    gl_FragColor = vec4(rgb,1);
}
""")

FS_point = """
#version 120

varying float v_z;

void main() {
    vec3 rgb=mix(vec3(1,0.5,0),vec3(0,0.5,1.0),v_z);
    gl_FragColor = vec4(rgb,1);
}
"""


class Canvas(app.Canvas):

    def __init__(self, surface):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface")
        gloo.set_state(clear_color=(0, 0, 0, 1), depth_test=False, blend=False)
        self.program = gloo.Program(VS, FS_triangle)
        self.program["a_position"] = surface.position()
        self.program["a_height"] = surface.get_heights_in_norm_coords()

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
        #self.program.draw('points')
        self.program.draw('triangles', self.triangles)

    def on_timer(self, event):
        self.t += 0.05
        self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def on_mouse_press(self, event):
        surface.one_random_wave()

if __name__ == '__main__':
    surface = NaturalWaves(size=(20, 20), max_height=0.9)
    surface.generate_random_waves(intensity=1)
    c = Canvas(surface)
    app.run()
