#version 100

in vec2 pos;


uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(pos, 0.0, 1.0);
}
