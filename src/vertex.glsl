#version 100

in vec2 pos;
in vec4 color0;
uniform vec4 color;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(pos, 0.0, 1.0);
}
