#version 100
precision mediump float;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(600.0, 600.0); // Adjust for your resolution
    uv -= 0.5;

    vec3 bgColor = mix(vec3(0.05, 0.05, 0.1), vec3(0.1, 0.1, 0.2), uv.y + 0.5);

    gl_FragColor = vec4(bgColor, 1.0);

}