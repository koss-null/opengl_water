VS = ("""
#version 120

uniform float u_eye_height;
uniform vec2  u_eye_position;

uniform vec3 u_main_x;
uniform vec3 u_main_y;
uniform vec3 u_main_z;

uniform float test;

attribute vec2  a_position;
attribute float a_height;
attribute vec3  a_normal;
attribute float a_bed_depth;

varying float v_bed_depth;
varying vec3 v_normal;
varying vec3 v_position;
varying vec3 eye_position;

void main (void) {
    v_normal = normalize(a_normal);
    v_bed_depth = a_bed_depth;
    vec3 rotation = normalize(vec3(u_main_x.x * u_main_y.x * u_main_z.x,
                                u_main_x.y * u_main_y.y * u_main_z.y,
                                u_main_x.z * u_main_y.z * u_main_z.z));
                                
    v_position = vec3(a_position.xy, a_height);
    vec3 eye = vec3(u_eye_position.xy, u_eye_height);
    eye_position = normalize(vec3(0.5, 0.5, u_eye_height));

    float z = (1-(1+a_height)/(1+u_eye_height));

    gl_Position = vec4(a_position.xy/2, a_height * z, z);
}
""")

FS_triangle = ("""
#version 120

uniform sampler2D u_sky_texture;
uniform sampler2D u_bed_texture;
uniform sampler2D u_shademap_texture;

uniform float test;

uniform vec3 u_sun_direction;
uniform vec3 u_sun_color;
uniform vec3 u_ambient_color;

uniform float u_alpha;
uniform vec2  u_eye_position;
uniform float u_eye_height;

uniform int u_show_bed;
uniform int u_show_sky;

varying float v_bed_depth;

varying vec3 v_normal;
varying vec3 v_position;
varying vec3 eye_position;

void main() {    
    //////////////// sky color
    vec3 eye = normalize(vec3(vec2(0, 0).xy, u_eye_height));
    vec3 from_eye = -normalize(v_position - eye);
    vec3 normal = normalize(v_normal);
    
    vec3 reflected = normalize(from_eye - 2 * normal * dot(normal,from_eye));
    vec2 sky_texcoord = 0.5 * reflected.xy/reflected.z + vec2(0.5,0.5);
    
    vec3 sky_color = texture2D(u_sky_texture, sky_texcoord).rgb; //vec3(0.5, 0.3, 0.6);

    /////////////// bed color

    vec3 cr = cross(v_normal, from_eye);
    float d = 1 - u_alpha*u_alpha*dot(cr,cr);
    float c2 = sqrt(d);
    vec3 refracted = normalize(u_alpha*cross(cr, v_normal) - v_normal*c2);

    float c1 = -dot(v_normal, from_eye);
    float t = (-v_bed_depth-v_position.z)/refracted.z;
    vec3 point_on_bed = v_position + t * refracted;
    vec2 bed_texcoord = point_on_bed.xy + vec2(0.5,0.5);
    vec3 bed_color = texture2D(u_bed_texture, bed_texcoord).rgb; //vec3(0.1, 0.3, 0.7);

    float cos_phi = dot(reflected, eye) /
                        length(reflected) /
                        length(eye);
    
    //////////////////

    vec3 k_amb = u_ambient_color;
    vec3 k_spec = u_sun_color;
    vec3 k_def = vec3(0, 0, 0);
    
    if (u_show_sky == 0) {
        k_def = k_def + sky_color * cos_phi * cos_phi;
    }
    if (u_show_bed == 0) {
        k_def = k_def + bed_color * (1 - cos_phi * cos_phi);
    }
    k_def = k_def / 2;
    
    vec3 image_clr = k_amb + k_def * dot(v_normal, u_sun_direction) + k_spec * dot(v_normal, from_eye);

    gl_FragColor.rgb = clamp(image_clr, 0.0, 1.0);
    gl_FragColor.a = 1;
}
""")

FS_point = """
#version 120

varying float v_z;

void main() {
    vec3 rgb=mix(vec3(1,0.5,0), vec3(0,0.5,1.0),v_z);
    gl_FragColor = vec4(shade_color, 1);
}
"""