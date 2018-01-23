VS = ("""
#version 120

uniform float test;

uniform float     u_eye_height;
uniform vec2      u_eye_position;
uniform vec3      u_angle;
uniform vec2      u_surf_size;
uniform sampler2D u_height;
uniform sampler2D u_bed_depth;

attribute vec2  a_position;
attribute float a_dot;

varying float v_dot;
varying vec3  v_normal;
varying float v_bed_depth;
varying vec3  v_position;
varying vec3  eye_position;
varying mat3  v_mat;

mat3 get_matrix() {
    mat3 mat;
    float A = cos(u_angle.x);
    float B = sin(u_angle.x);
    float C = cos(u_angle.y);
    float D = sin(u_angle.y);
    float E = cos(u_angle.z);
    float F = sin(u_angle.z);

    float AD =   A * D;
    float BD =   B * D;

    mat[0][0] = C * E;
    mat[0][1] = -C * F;
    mat[0][2] = -D;
    mat[1][0] = -BD * E + A * F;
    mat[1][1] = BD * F + A * E;
    mat[1][2] = -B * C;
    mat[2][0] = AD * E + B * F;
    mat[2][1] = -AD * F + B * E;
    mat[2][2] =  A * C;
    
    return mat;
}

vec3 get_normal(vec2 position) {
    vec3 heights;
    float dev = 100;
    
    heights = texture2D(u_height, position).rgb;

    float z1 = heights.x;
    float z2 = heights.y;
    float z3 = heights.z;

    float A = z2 - z1;
    float B = z2 - z3;
    float C = -1;

    return(-normalize(vec3(A, B, C)));
}

void main (void) {
    v_mat = get_matrix();
    v_dot = a_dot;

    vec2 texture_position = vec2((a_position.x + 1) * 0.5, (a_position.y + 1) * 0.5);
    vec3 normal = get_normal(texture_position);
    float height;
    
    if (a_dot == 1) {
        height = texture2D(u_height, texture_position).rgb.x;
        v_bed_depth = texture2D(u_bed_depth, texture_position).rgb.x;
    } else if (a_dot == 2) {
        height = texture2D(u_height, texture_position).rgb.y;
        v_bed_depth = texture2D(u_bed_depth, texture_position).rgb.y;
    }
    v_normal = normalize(normal);
    vec3 mat_normal = v_mat * v_normal;
    
    v_position = vec3(a_position.xy, height);                                
    vec3 mat_position = v_mat * v_position;
    vec3 eye = v_mat * vec3(u_eye_position.xy, u_eye_height);
    eye_position = normalize(vec3(a_position.xy, u_eye_height));

    float z = (1 - (1 + height)/(1 + u_eye_height));

    gl_Position = vec4(mat_position.xy/2, mat_position.z * z, z);   
}
""")

FS_triangle = ("""
#version 120

uniform sampler2D u_sky_texture;
uniform sampler2D u_bed_texture;
uniform sampler2D u_bed_depth;

uniform float test;

uniform vec3 u_sun_direction;
uniform vec3 u_sun_color;
uniform vec3 u_ambient_color;
uniform vec3 u_angle;

uniform float u_alpha;
uniform vec2  u_eye_position;
uniform float u_eye_height;

uniform int u_show_bed;
uniform int u_show_sky;

varying float v_dot;
varying vec3  v_normal;
varying float v_bed_depth;
varying vec3  v_h;
varying vec3  v_position;
varying vec3  eye_position;
varying mat3  v_mat;

vec3 get_normal(vec2 position) {
    vec3 heights;
    float dev = 100;
    
    heights = texture2D(u_bed_depth, position).rgb;

    float z1 = heights.x;
    float z2 = heights.y;
    float z3 = heights.z;

    float A = z2 - z1;
    float B = z2 - z3;
    float C = -1;

    return(-normalize(vec3(A, B, C)));
}

void main() {
     //gl_FragColor.rgb = clamp(v_normal, 0.0, 1.0);
     //return;
    //////////////// sky color
    vec3 position = v_mat * v_position;
    
    vec3 eye = v_mat * vec3(u_eye_position.xy, u_eye_height);
    vec3 from_eye = normalize(v_position - eye);
    vec3 normal = v_mat * v_normal;
    
    vec3 reflected = normalize(from_eye - 2 * v_normal * dot(v_normal, from_eye));
    vec2 sky_texcoord = 0.5 * reflected.xy/reflected.z + vec2(0.5,0.5);
    
    vec3 sky_color = texture2D(u_sky_texture, sky_texcoord).rgb; //vec3(0.5, 0.3, 0.6);

    from_eye = from_eye * v_mat;

    /////////////// bed color

    vec3 texture_position = vec3((v_position.x), (v_position.y), v_position.z);
    
    float bed_depth = v_bed_depth;
    
    vec3 cr = cross(normal, from_eye);
    float d = 1 - u_alpha*u_alpha*dot(cr,cr);
    float c2 = sqrt(d);
    vec3 refracted = normalize(u_alpha*cross(cr, normal) - normal*c2);

    float c1 = -dot(normal, from_eye);
    float t = (-bed_depth - position.z)/refracted.z;
    vec3 point_on_bed = v_mat * (texture_position + t * refracted);
    vec2 bed_texcoord = point_on_bed.xy + vec2(0.5,0.5);

    float diw = length(point_on_bed - position);
    vec3 filter = vec3(1, 0.5, 0.3);
    vec3 v_mask = vec3(exp(-diw * filter.x), exp(-diw * filter.y), exp(-diw * filter.z));

    vec3 bed_color = texture2D(u_bed_texture, bed_texcoord).rgb * v_mask; //vec3(0.1, 0.3, 0.7);

    float cos_phi = dot(reflected, from_eye) /
                        length(reflected) /
                        length(from_eye);
    
    //////////////////

    vec3 k_amb = u_ambient_color;
    vec3 k_spec = u_sun_color;
    vec3 k_def = vec3(1, 0.8, 1);
    
    float reflectance_s = pow((u_alpha*c1-c2)/(u_alpha*c1+c2), 3);
    float reflectance_p = pow((u_alpha*c2-c1)/(u_alpha*c2+c1), 3);
    float reflectance = (reflectance_s + reflectance_p) / 2;
    
    float reflected_intensity = pow(cos_phi, 1) ;
    
    float sky_coef = 0.6;
    float bed_coef = 1;
    
    if (cos_phi > 0) {
        vec3 cash = sky_color;
        sky_color = bed_color;
        bed_color = cash;
        sky_coef = 1;
        bed_coef = 0.6;
    }    
    
    if (u_show_sky == 0) {
        k_def = k_def + sky_color * (1 - reflected_intensity) * sky_coef;
    }
    if (u_show_bed == 0) {
        k_def = k_def + bed_color * (reflected_intensity) * bed_coef;
    }
    
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