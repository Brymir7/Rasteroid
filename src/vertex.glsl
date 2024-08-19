#version 100

in vec2 pos;
in vec4 color0;
uniform vec2 screen_size;
uniform vec2 shake_offset;
uniform mat4 mvp;

varying vec4 color;
void main() {
    vec4 position = mvp * vec4(pos, 0.0, 1.0);
    position.xy += shake_offset / screen_size;
    gl_Position = position;
    color = color0;
}
