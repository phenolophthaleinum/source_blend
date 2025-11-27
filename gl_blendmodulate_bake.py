from dataclasses import dataclass
from typing import ClassVar


@dataclass
class ShaderVariables:
    """Shader uniform and attribute variable names."""
    in_vert: ClassVar[str] = "in_vert"
    in_text: ClassVar[str] = "in_text"
    topTexture: ClassVar[str] = "topTexture"
    bottomTexture: ClassVar[str] = "bottomTexture"
    rComponent: ClassVar[str] = "rComponent"
    alpha: ClassVar[str] = "alpha"
    input_black: ClassVar[str] = "input_black"
    input_white: ClassVar[str] = "input_white"
    output_black: ClassVar[str] = "output_black"
    output_white: ClassVar[str] = "output_white"


class SubstractShader:
    """OpenGL shader for blend modulate baking."""
    
    variables = ShaderVariables
    
    vertex_shader = """
#version 330
in vec2 in_vert;
in vec2 in_text;
out vec2 tex_coord;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    tex_coord = in_text;
}
"""
    
    fragment_shader = """
#version 330
uniform sampler2D topTexture;
uniform sampler2D bottomTexture;
uniform sampler2D rComponent;
uniform float alpha;

uniform float input_black;
uniform float input_white;
uniform float output_black;
uniform float output_white;

in vec2 tex_coord;
out vec4 fragColor;

const float GAMMA = 2.2;
const float LEVEL_GAMMA = 1.0;
                                
vec3 srgb_to_linear(vec3 color) {
    return pow(color, vec3(GAMMA));
}

vec3 linear_to_srgb(vec3 color) {
    return pow(color, vec3(1.0 / GAMMA));
}

float adjust_pixel(float x) {
    if (input_black >= input_white) return x;
    float normalised = (x - input_black) / (input_white - input_black);
    normalised = clamp(normalised, 0.0, 1.0);
    float gamma_adjusted = pow(normalised, LEVEL_GAMMA);
    float final_pixel = gamma_adjusted * (output_white - output_black) + output_black;
    return clamp(final_pixel, 0.0, 1.0);
}

vec3 adjust_color_levels(vec3 color) {
    return vec3(
        adjust_pixel(color.r),
        adjust_pixel(color.g),
        adjust_pixel(color.b)
    );
}

void main() {
    vec4 topTex = texture(topTexture, tex_coord);
    vec4 bottomTex = texture(bottomTexture, tex_coord);
    vec4 rTex = texture(rComponent, tex_coord);
    vec3 topLinear = srgb_to_linear(topTex.rgb);
    vec3 bottomLinear = srgb_to_linear(bottomTex.rgb);

    float effectiveAlpha = topTex.a * alpha;
    vec3 blendedLinear = bottomLinear - topLinear * effectiveAlpha;
    blendedLinear = clamp(blendedLinear, 0.0, 1.0);

    vec3 blendedSrgb = linear_to_srgb(blendedLinear);
    float compositeAlpha = effectiveAlpha + bottomTex.a * (1.0 - effectiveAlpha);
    
    vec4 blendModulate = vec4(blendedSrgb, compositeAlpha);
    vec3 finalBlendModulate = adjust_color_levels(blendModulate.rgb);

    fragColor = vec4(rTex.r, finalBlendModulate.g, 0, 1.0);
}
"""

class DivideShader:
    """OpenGL shader for blend modulate baking."""
    
    variables = ShaderVariables
    
    vertex_shader = """
#version 330
in vec2 in_vert;
in vec2 in_text;
out vec2 tex_coord;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    tex_coord = in_text;
}
"""
    
    fragment_shader = """
#version 330
uniform sampler2D topTexture;
uniform sampler2D bottomTexture;
uniform sampler2D rComponent;
uniform float alpha;

uniform float input_black;
uniform float input_white;
uniform float output_black;
uniform float output_white;

in vec2 tex_coord;
out vec4 fragColor;

const float GAMMA = 2.2;
const float LEVEL_GAMMA = 1.0;
                                
vec3 srgb_to_linear(vec3 color) {
    return pow(color, vec3(GAMMA));
}

vec3 linear_to_srgb(vec3 color) {
    return pow(color, vec3(1.0 / GAMMA));
}

float adjust_pixel(float x) {
    if (input_black >= input_white) return x;
    float normalised = (x - input_black) / (input_white - input_black);
    normalised = clamp(normalised, 0.0, 1.0);
    float gamma_adjusted = pow(normalised, LEVEL_GAMMA);
    float final_pixel = gamma_adjusted * (output_white - output_black) + output_black;
    return clamp(final_pixel, 0.0, 1.0);
}

vec3 adjust_color_levels(vec3 color) {
    return vec3(
        adjust_pixel(color.r),
        adjust_pixel(color.g),
        adjust_pixel(color.b)
    );
}

void main() {
    vec4 topTex = texture(topTexture, tex_coord);
    vec4 bottomTex = texture(bottomTexture, tex_coord);
    vec4 rTex = texture(rComponent, tex_coord);
    vec3 topLinear = srgb_to_linear(topTex.rgb);
    vec3 bottomLinear = srgb_to_linear(bottomTex.rgb);

    float effectiveAlpha = topTex.a * alpha;
    vec3 blendedLinear = bottomLinear / topLinear * effectiveAlpha;
    blendedLinear = clamp(blendedLinear, 0.0, 1.0);

    vec3 blendedSrgb = linear_to_srgb(blendedLinear);
    float compositeAlpha = effectiveAlpha + bottomTex.a * (1.0 - effectiveAlpha);
    
    vec4 blendModulate = vec4(blendedSrgb, compositeAlpha);
    vec3 finalBlendModulate = adjust_color_levels(blendModulate.rgb);

    fragColor = vec4(rTex.r, finalBlendModulate.g, 0, 1.0);
}
"""


BLEND_SHADERS = {
    "substract": SubstractShader,
    "divide": DivideShader,
}