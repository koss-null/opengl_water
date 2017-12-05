VS = ("""
#version 120

attribute vec2 a_position;
attribute vec3 a_normal;
attribute float a_height;

uniform vec3 u_sun_direction;

varying float v_directed_light;
varying vec3 v_normal;

void main (void) {
    vec3 v_normal = normalize(a_normal);
    v_directed_light = max(0, -dot(v_normal, u_sun_direction));

    gl_Position = vec4(a_position.xy, a_height * a_height, a_height);
}
""")

FS_triangle = ("""
#version 120

uniform vec3 u_sun_color;
uniform vec3 u_ambient_color;

varying float v_directed_light;
varying vec3 v_normal;

void main() {
    vec3 rgb = clamp(u_sun_color*v_directed_light + u_ambient_color, 0.0, 1.0);

    gl_FragColor = vec4(rgb, 1);
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