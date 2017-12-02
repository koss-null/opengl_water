VS = ("""
#version 120

attribute vec2 a_position;
attribute float a_height;

varying float v_z;

void main (void) {
    v_z = a_height;
    gl_Position = vec4(a_position.xy, a_height, a_height);
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