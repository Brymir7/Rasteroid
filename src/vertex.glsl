#version 100

in vec2 pos;
in vec4 color0;

uniform mat4 mvp;

varying vec4 color;
void main() {
    gl_Position = mvp * vec4(pos, 0.0, 1.0);
    color = color0;
}
