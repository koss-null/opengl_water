VS = ("""
#version 120

uniform float u_eye_height;
uniform vec2  u_eye_position;

uniform vec3 u_main_x;
uniform vec3 u_main_y;
uniform vec3 u_main_z;

attribute vec2  a_position;
attribute float a_height;
attribute vec3  a_normal;

varying vec3 v_normal;
varying vec3 v_position;
varying vec3 eye_position;

void main (void) {
    v_normal = normalize(a_normal);
    vec3 rotation = normalize(vec3(u_main_x.x * u_main_y.x * u_main_z.x,
                                u_main_x.y * u_main_y.y * u_main_z.y,
                                u_main_x.z * u_main_y.z * u_main_z.z));
                                
    v_position = vec3(a_position.xy, a_height);
    vec3 eye = vec3(u_eye_position.xy, u_eye_height);
    eye_position = normalize(vec3(0.5, 0.5, u_eye_height));

    float cos_phi = dot(v_normal, -eye) / length(v_normal) / length(eye);

    float z = (1-(1+a_height)/(1+u_eye_height));

    gl_Position = vec4(a_position.xy/2, a_height * z, z);
}
""")

FS_triangle = ("""
#version 120

uniform sampler2D u_sky_texture;
uniform sampler2D u_bed_texture;
uniform sampler2D u_shademap_texture;

uniform vec3 u_sun_direction;
uniform vec3 u_sun_color;
uniform vec3 u_ambient_color;

uniform float u_alpha;
uniform float u_bed_depth;
uniform vec2  u_eye_position;
uniform float u_eye_height;

varying vec3 v_normal;
varying vec3 v_position;
varying vec3 eye_position;

void main() {    
    //// useless

    vec3 V = normalize(vec3(v_normal.x, -(pow(v_normal.x, 2) + pow(v_normal.z, 2))/v_normal.y, v_normal.z));
    vec3 H = normalize(vec3(v_normal.x, v_normal.y, -(pow(v_normal.x, 2) + pow(v_normal.y, 2))/v_normal.z));
    vec3 norm_sun_direction = normalize(u_sun_direction);

    float u = (dot(V, norm_sun_direction) * 128 + 127) / 255;
    float v = (dot(H, norm_sun_direction) * 128 + 127) / 255;

    vec2 shade_coords = vec2(u, v);
    vec3 shade_color = texture2D(u_shademap_texture, shade_coords).rgb;

    float luminance = pow((0.299*shade_color.x + 0.587*shade_color.y + 0.114*shade_color.z), 1);
    //if (luminance > 1) {
    //    luminance = 1;
    //}
    /////////////////// useless_end

    vec3 eye = vec3(u_eye_position.xy, u_eye_height);
    vec3 from_eye = -normalize(v_position - eye);

    vec3 reflected = normalize(from_eye - 2*v_normal*dot(v_normal, from_eye));

    vec2 sky_texcoord = 0.05 * reflected.xy / (reflected.z + 1) + vec2(0.5, 0.5);
    vec3 sky_color = texture2D(u_sky_texture, sky_texcoord).rgb; //vec3(0.5, 0.3, 0.6);

    /////////////// bed color

    vec3 cr = cross(v_normal, from_eye);
    float d = 1 - u_alpha*u_alpha*dot(cr,cr);
    float c2 = sqrt(d);
    vec3 refracted = normalize(u_alpha*cross(cr, v_normal) - v_normal*c2);

    float c1 = -dot(v_normal, from_eye);
    float t = (-u_bed_depth-v_position.z)/refracted.z;
    vec3 point_on_bed = v_position + t * refracted;
    vec2 bed_texcoord = point_on_bed.xy + vec2(0.5,0.5);
    vec3 bed_color = texture2D(u_bed_texture, bed_texcoord).rgb; //vec3(0.1, 0.3, 0.7);

    float cos_phi = dot(v_normal, eye) /
                        sqrt(pow(v_normal.x, 2) + pow(v_normal.y, 2) + pow(v_normal.z, 2)) /
                        sqrt(pow(eye.x, 2) + pow(eye.y, 2) + pow(eye.z, 2));
    
    //////////////////

    vec3 k_amb = u_ambient_color;
    vec3 k_spec = u_sun_color;
    vec3 k_def = (sky_color * (1 - cos_phi * cos_phi) + bed_color * cos_phi * cos_phi) * 0.7;
    
    vec3 image_clr = k_amb + k_def * dot(v_normal, u_sun_direction) + k_spec * dot(v_normal, eye_position);

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