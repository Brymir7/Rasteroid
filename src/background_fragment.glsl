uniform float u_time;

// Improved hash function for better randomness
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// Smooth noise function
float noise(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(dot(hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
                   dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
               mix(dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
                   dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x), u.y);
}

// Star field function
float starField(vec2 uv, float threshold) {
    float n = noise(uv *500.0);
    float stars = smoothstep(threshold, threshold + 0.05, n);
    return stars;
}

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(800.0, 600.0); // Adjust for your resolution
    uv -= 0.5;
    uv.x *= 800.0 / 600.0; // Aspect ratio correction
    
    // Create a more noticeable flowing movement
    vec2 movement = vec2(u_time * 0.02, sin(u_time * 0.01) * 0.05);
    uv += movement;
    
    // Generate multiple layers of stars with different densities and sizes
    float stars1 = starField(uv, 0.75) * 0.5;
    float stars2 = starField(uv * 2.0 + 1000.0, 0.8) * 0.3;
    float stars3 = starField(uv * 4.0 + 2000.0, 0.85) * 0.2;
    
    // Combine star layers
    float stars = stars1 + stars2 + stars3;
    
    // Add a subtle color gradient
    vec3 bgColor = mix(vec3(0.05, 0.05, 0.1), vec3(0.1, 0.1, 0.2), uv.y + 0.5);
    
    // Create a more pronounced pulsing effect
    float pulse = sin(u_time * 0.5) * 0.1 + 9.9;
    stars *= 0.7 + pulse * 0.3;
    
    // Final color
    vec3 color = bgColor + stars * vec3(1.0, 0.9, 0.7);
    
    gl_FragColor = vec4(color, 1.0);

}