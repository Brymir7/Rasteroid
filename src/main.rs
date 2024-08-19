use conf::Conf;
use miniquad::*;
use glam::{ vec3, Mat4, Quat, Vec2 };
use rand::Rng;
use std::collections::HashSet;

const WORLD_WIDTH: f32 = 1200.0;
const WORLD_HEIGHT: f32 = 800.0;
const PHYSICS_FRAME_TIME: f32 = 1.0 / 60.0;
const PLAYER_HITBOX_SIZE: f32 = 10.0;
const ASTEROID_BASE_SPEED: f32 = 50.0;
const DEFAULT_DASH_FRAMES: u8 = 20;
const DEFAULT_DASH_SPEED: f32 = 3.0;

struct Asteroids {
    positions: Vec<Vec2>,
    velocities: Vec<Vec2>,
    rotations: Vec<f32>,
    health: Vec<u8>,
    base_healths: Vec<u8>,
    collision: Vec<Collision>,
    render_data: Vec<RenderData>,
}
struct Bullets {
    positions: Vec<Vec2>,
    velocities: Vec<Vec2>,
    rotations: Vec<f32>,
    health: Vec<u8>,
    collision: Vec<Collision>,
    render_data: Vec<RenderData>,
}
struct Player {
    pos: Vec2,
    vel: Vec2,
    rot: f32,
    health: u8,
    collision: Collision,
    render_data: RenderData,
    is_dashing: u8, // amount of physics frames he is dashing
}
struct RenderDataCreator;
impl RenderDataCreator {
    fn create_triangle_render_data(ctx: &mut Context, pipeline_handle: u8) -> RenderData {
        let vertices = vec![Vec2::new(-0.5, 0.25), Vec2::new(0.5, 0.25), Vec2::new(0.0, -1.0)];
        Self::create_render_data(ctx, &vertices, &[0, 1, 2], pipeline_handle)
    }

    fn create_asteroid_render_data(ctx: &mut Context, pipeline_handle: u8) -> RenderData {
        let mut rng = rand::thread_rng();
        let num_points = 8; // Number of varying points for shape
        let angle_increment = (2.0 * std::f32::consts::PI) / (num_points as f32);

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for i in 0..num_points {
            let angle = (i as f32) * angle_increment;
            let radius_variation: f32 = rng.gen_range(0.8..1.2);
            let x = radius_variation * angle.cos();
            let y = radius_variation * angle.sin();

            vertices.push(Vec2 { x, y });
        }
        for i in 0..num_points {
            indices.push(i as u16);
            indices.push(((i + 1) % num_points) as u16);
        }

        Self::create_render_data(ctx, &vertices, &indices, pipeline_handle)
    }

    fn create_bullet_render_data(ctx: &mut Context, pipeline_handle: u8) -> RenderData {
        let vertices = vec![
            Vec2::new(-0.1, 0.2),
            Vec2::new(0.1, 0.2),
            Vec2::new(0.1, -0.2),
            Vec2::new(-0.1, -0.2)
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        Self::create_render_data(ctx, &vertices, &indices, pipeline_handle)
    }

    fn create_render_data(
        ctx: &mut Context,
        vertices: &[Vec2],
        indices: &[u16],
        pipeline_handle: u8
    ) -> RenderData {
        let mut vertex_buffer_data: Vec<f32> = Vec::new();
        for vertex in vertices {
            vertex_buffer_data.push(vertex.x);
            vertex_buffer_data.push(vertex.y);
            vertex_buffer_data.extend_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        }

        let vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&vertex_buffer_data)
        );

        let index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(indices)
        );

        RenderData {
            indices: indices.to_vec(),
            vertex_buffer,
            index_buffer,
            pipeline_handle,
        }
    }
}
struct CameraShake {
    duration: f32,
    intensity: f32,
    current_time: f32,
}

impl CameraShake {
    fn new(duration: f32, intensity: f32) -> Self {
        Self {
            duration,
            intensity,
            current_time: 0.0,
        }
    }

    fn update(&mut self, dt: f32) -> Vec2 {
        if self.current_time >= self.duration {
            return Vec2::ZERO;
        }
        self.current_time += dt;
        let progress = self.current_time / self.duration;
        let damping = 1.0 - progress;

        let angle = rand::random::<f32>() * std::f32::consts::TAU;
        let offset = Vec2::new(angle.cos(), angle.sin()) * self.intensity * damping;
        offset
    }
}
enum WorldEvent {
    CameraShake,
    PlayerDeath,
}
struct World {
    asteroids: Asteroids,
    bullets: Bullets,
    player: Player,
    pipelines: Vec<Pipeline>, // refactor into static size array, when all pipelines are known
    camera_shake: Option<CameraShake>,
}

impl World {
    fn new(ctx: &mut Context, pipelines: Vec<Pipeline>) -> Self {
        World {
            asteroids: Asteroids {
                positions: Vec::new(),
                velocities: Vec::new(),
                rotations: Vec::new(),
                health: Vec::new(),
                base_healths: Vec::new(),
                collision: Vec::new(),
                render_data: Vec::new(),
            },
            bullets: Bullets {
                positions: Vec::new(),
                velocities: Vec::new(),
                rotations: Vec::new(),
                health: Vec::new(),
                collision: Vec::new(),
                render_data: Vec::new(),
            },
            player: Player {
                pos: Vec2::new(WORLD_WIDTH / 2.0, WORLD_HEIGHT / 2.0),
                vel: Vec2::new(0.0, 0.0),
                rot: 0.0,
                health: 3,
                collision: Collision { radius: PLAYER_HITBOX_SIZE },
                render_data: RenderDataCreator::create_triangle_render_data(ctx, 0),
                is_dashing: 0,
            },
            pipelines,
            camera_shake: None,
        }
    }
    fn add_bullet(&mut self, ctx: &mut Context, pos: Vec2, vel: Vec2, rot: f32) {
        self.bullets.positions.push(pos);
        self.bullets.velocities.push(vel);
        self.bullets.rotations.push(rot);
        self.bullets.health.push(1);
        self.bullets.collision.push(Collision { radius: 10.0 });
        self.bullets.render_data.push(RenderDataCreator::create_bullet_render_data(ctx, 0));
    }
    fn add_asteroid(&mut self, ctx: &mut Context, pos: Vec2, vel: Vec2, rot: f32, health: u8) {
        self.asteroids.positions.push(pos);
        self.asteroids.velocities.push(vel);
        self.asteroids.rotations.push(rot);
        self.asteroids.health.push(health);
        self.asteroids.collision.push(Collision { radius: 30.0 * (health as f32) });
        self.asteroids.base_healths.push(health);
        self.asteroids.render_data.push(RenderDataCreator::create_asteroid_render_data(ctx, 1));
    }
    fn remove_object_by_identifier(&mut self, identifier: &ObjectIdentifier) {
        match identifier.object_type {
            EntityType::Asteroid => {
                // we just need to upkeep positions[i] == vel[i] == rot[i] == health[i] == collision[i] == render_data[i]
                if self.asteroids.positions.len() <= identifier.idx_original_array {
                    // already removed
                    return;
                }
                self.asteroids.positions.swap_remove(identifier.idx_original_array);
                self.asteroids.velocities.swap_remove(identifier.idx_original_array);
                self.asteroids.rotations.swap_remove(identifier.idx_original_array);
                self.asteroids.health.swap_remove(identifier.idx_original_array);
                self.asteroids.collision.swap_remove(identifier.idx_original_array);
                self.asteroids.render_data.swap_remove(identifier.idx_original_array);
            }
            EntityType::Bullet => {
                // we just need to upkeep positions[i] == vel[i] == rot[i] == health[i] == collision[i] == render_data[i]
                if self.bullets.positions.len() <= identifier.idx_original_array {
                    // already removed
                    return;
                }
                self.bullets.positions.swap_remove(identifier.idx_original_array);
                self.bullets.velocities.swap_remove(identifier.idx_original_array);
                self.bullets.rotations.swap_remove(identifier.idx_original_array);
                self.bullets.health.swap_remove(identifier.idx_original_array);
                self.bullets.collision.swap_remove(identifier.idx_original_array);
                self.bullets.render_data.swap_remove(identifier.idx_original_array);
            }
            _ => {
                panic!("Cannot remove player!");
            }
        }
    }
    fn handle_collision_events(
        &mut self,
        ctx: &mut Context,
        events: Vec<CollisionEvent>
    ) -> Vec<WorldEvent> {
        let mut world_events = Vec::new();
        for event in events {
            match event.event_type {
                CollisionEventType::PlayerHit => {
                    if self.asteroids.health.len() <= event.triggered_by.idx_original_array {
                        // already removed
                        continue;
                    }
                    self.remove_object_by_identifier(&event.triggered_by);
                    self.player.health = self.player.health.saturating_sub(1);
                    if self.player.health == 0 {
                        world_events.push(WorldEvent::PlayerDeath);
                        continue;
                    }
                    world_events.push(WorldEvent::CameraShake);
                }
                CollisionEventType::AsteroidHit => {
                    if self.asteroids.health.len() <= event.target.idx_original_array {
                        // already removed
                        continue;
                    }

                    let original_health =
                        self.asteroids.base_healths[event.target.idx_original_array];
                    let pos = self.asteroids.positions[event.target.idx_original_array];

                    if self.asteroids.health[event.target.idx_original_array] < 2 {
                        self.remove_object_by_identifier(&event.target);
                        if original_health <= 1 {
                            continue;
                        }
                        for _ in 0..original_health {
                            let vel = Vec2::new(
                                rand::thread_rng().gen_range(-1.0..1.0) * ASTEROID_BASE_SPEED,
                                rand::thread_rng().gen_range(-1.0..1.0) * ASTEROID_BASE_SPEED
                            );
                            let rot = rand::thread_rng().gen_range(0.0..std::f32::consts::PI * 2.0);
                            if original_health - 1 == 1 {
                                // emtpy range sample otherwise
                                self.add_asteroid(ctx, pos, vel, rot, 1);
                                continue;
                            }
                            let new_size = rand::thread_rng().gen_range(1..original_health - 1);
                            self.add_asteroid(ctx, pos, vel, rot, new_size);
                        }
                    } else {
                        self.asteroids.health[event.target.idx_original_array] -= 1; // ignore underflow because we already checked for 1
                    }
                }
                CollisionEventType::BulletHit => {
                    if self.bullets.health.len() <= event.target.idx_original_array {
                        // already removed
                        continue;
                    }
                    if self.bullets.health[event.target.idx_original_array] < 2 {
                        self.remove_object_by_identifier(&event.target);
                        continue;
                    }
                    self.bullets.health[event.target.idx_original_array] -= 1; // ignore underflow because we already checked for 1
                }
            }
        }
        world_events
    }

    fn update(&mut self, ctx: &mut Context) {
        let out_of_bounds = MovementSystem::update(
            &mut self.asteroids.positions,
            &self.asteroids.velocities
        );
        for event in out_of_bounds {
            self.remove_object_by_identifier(
                &(ObjectIdentifier {
                    idx_original_array: event.index,
                    object_type: EntityType::Asteroid,
                })
            );
        }
        let out_of_bounds = MovementSystem::update(
            &mut self.bullets.positions,
            &self.bullets.velocities
        );
        for event in out_of_bounds {
            self.remove_object_by_identifier(
                &(ObjectIdentifier {
                    idx_original_array: event.index,
                    object_type: EntityType::Bullet,
                })
            );
        }
        if self.player.is_dashing > 0 {
            self.player.is_dashing -= 1;
            self.player.pos += self.player.vel * DEFAULT_DASH_SPEED * PHYSICS_FRAME_TIME;
        } else {
            self.player.pos += self.player.vel * PHYSICS_FRAME_TIME;
        }
        if self.player.pos.x < 0.0 {
            self.player.pos.x = WORLD_WIDTH;
        } else if self.player.pos.x > WORLD_WIDTH {
            self.player.pos.x = 0.0;
        }
        if self.player.pos.y < 0.0 {
            self.player.pos.y = WORLD_HEIGHT;
        } else if self.player.pos.y > WORLD_HEIGHT {
            self.player.pos.y = 0.0;
        }

        let world_events = self.handle_collision_events(
            ctx,
            CollisionSystem::run_asteroid_bullet_collision(
                &self.bullets.positions,
                &self.asteroids.positions,
                &self.bullets.collision
                    .iter()
                    .map(|c| c.radius)
                    .collect::<Vec<_>>(),
                &self.asteroids.collision
                    .iter()
                    .map(|c| c.radius)
                    .collect::<Vec<_>>()
            )
                .into_iter()
                .chain(
                    CollisionSystem::run_player_asteroid_collision(
                        &self.player.pos,
                        &self.asteroids.positions,
                        self.player.collision.radius,
                        &self.asteroids.collision
                            .iter()
                            .map(|c| c.radius)
                            .collect::<Vec<_>>()
                    )
                )
                .collect()
        );
        self.handle_world_events(world_events);
    }

    fn handle_world_events(&mut self, events: Vec<WorldEvent>) {
        for event in events {
            match event {
                WorldEvent::CameraShake => {
                    self.camera_shake = Some(CameraShake::new(0.5, 10.0));
                }
                WorldEvent::PlayerDeath => {
                    // println!("Player died!");
                    self.player.pos = Vec2::new(WORLD_WIDTH / 2.0, WORLD_HEIGHT / 2.0);
                    self.player.vel = Vec2::new(0.0, 0.0);
                    self.player.rot = 0.0;
                    self.player.health = 3;
                    self.player.is_dashing = 0;
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Collision {
    radius: f32,
}

#[derive(Clone)]
struct RenderData {
    indices: Vec<u16>,
    vertex_buffer: BufferId,
    index_buffer: BufferId,
    pipeline_handle: u8,
}
#[derive(Clone, Copy, Debug)]
enum EntityType {
    Player,
    Asteroid,
    Bullet,
}
struct OutOfBoundsEvent {
    index: usize,
}
// Systems
struct MovementSystem;
impl MovementSystem {
    fn update(positions: &mut Vec<Vec2>, velocities: &Vec<Vec2>) -> Vec<OutOfBoundsEvent> {
        let mut out_of_bounds_events = Vec::new();
        for (idx_pos, vel) in positions.iter_mut().enumerate().zip(velocities.iter()) {
            let (idx, pos) = idx_pos;
            *pos += *vel * PHYSICS_FRAME_TIME;
            if pos.x < 0.0 || pos.x > WORLD_WIDTH || pos.y < 0.0 || pos.y > WORLD_HEIGHT {
                out_of_bounds_events.push(OutOfBoundsEvent { index: idx });
            }
        }
        out_of_bounds_events
    }
}
#[derive(Debug)]
enum CollisionEventType {
    PlayerHit,
    AsteroidHit,
    BulletHit,
}
#[derive(Debug)]
struct ObjectIdentifier {
    idx_original_array: usize,
    object_type: EntityType,
}
#[derive(Debug)]
struct CollisionEvent {
    event_type: CollisionEventType,
    triggered_by: ObjectIdentifier,
    target: ObjectIdentifier,
}
struct CollisionSystem;

impl CollisionSystem {
    fn run_asteroid_bullet_collision(
        bullet_positions: &[Vec2],
        asteroid_positions: &[Vec2],
        bullet_radii: &[f32],
        asteroid_radii: &[f32]
    ) -> Vec<CollisionEvent> {
        let mut events = Vec::new();

        for (bullet_idx, &bullet_pos) in bullet_positions.iter().enumerate() {
            for (asteroid_idx, &asteroid_pos) in asteroid_positions.iter().enumerate() {
                let distance = bullet_pos.distance(asteroid_pos);
                if distance < bullet_radii[bullet_idx] + asteroid_radii[asteroid_idx] {
                    events.push(CollisionEvent {
                        event_type: CollisionEventType::AsteroidHit,
                        triggered_by: ObjectIdentifier {
                            idx_original_array: bullet_idx,
                            object_type: EntityType::Bullet,
                        },
                        target: ObjectIdentifier {
                            idx_original_array: asteroid_idx,
                            object_type: EntityType::Asteroid,
                        },
                    });
                    events.push(CollisionEvent {
                        event_type: CollisionEventType::BulletHit,
                        triggered_by: ObjectIdentifier {
                            idx_original_array: asteroid_idx,
                            object_type: EntityType::Asteroid,
                        },
                        target: ObjectIdentifier {
                            idx_original_array: bullet_idx,
                            object_type: EntityType::Bullet,
                        },
                    });
                }
            }
        }
        events
    }

    fn run_player_asteroid_collision(
        player_position: &Vec2,
        asteroid_positions: &[Vec2],
        player_radius: f32,
        asteroid_radii: &[f32]
    ) -> Vec<CollisionEvent> {
        let mut events = Vec::new();

        for (asteroid_idx, &asteroid_pos) in asteroid_positions.iter().enumerate() {
            let distance = player_position.distance(asteroid_pos);
            if distance < player_radius + asteroid_radii[asteroid_idx] {
                events.push(CollisionEvent {
                    event_type: CollisionEventType::PlayerHit,
                    triggered_by: ObjectIdentifier {
                        idx_original_array: asteroid_idx,
                        object_type: EntityType::Asteroid,
                    },
                    target: ObjectIdentifier {
                        idx_original_array: 0, // Assuming player is always index 0
                        object_type: EntityType::Player,
                    },
                });
            }
        }
        events
    }
}
struct RenderSystem {}

impl RenderSystem {
    fn draw(
        ctx: &mut Context,
        positions: &[Vec2],
        rotations: &[f32],
        radii: &[f32],
        render_data: &[RenderData],
        entity_types: &[EntityType],
        pipelines: &[Pipeline],
        camera_shake: &mut Option<CameraShake>,
        dt: f32
    ) {
        let proj = Mat4::orthographic_rh_gl(0.0, WORLD_WIDTH, WORLD_HEIGHT, 0.0, -1.0, 1.0);
        let shake_offset = camera_shake.as_mut().map_or(Vec2::ZERO, |shake| shake.update(dt));
        for i in 0..positions.len() {
            let pos = positions[i];
            let rotation = rotations[i];
            let radius = radii[i];
            let render_data = &render_data[i];
            let entity_type = entity_types[i];

            let is_player = matches!(entity_type, EntityType::Player);

            let rotation_quat = Quat::from_rotation_z(rotation);
            let scale = if is_player {
                vec3(PLAYER_HITBOX_SIZE * 1.5, PLAYER_HITBOX_SIZE * 1.5, 1.0)
            } else {
                vec3(radius * 1.0, radius * 1.0, 1.0)
            };

            let model = Mat4::from_scale_rotation_translation(
                scale,
                rotation_quat,
                vec3(pos.x, pos.y, 0.0)
            );
            let mvp = proj * model;

            let bindings = Bindings {
                vertex_buffers: vec![render_data.vertex_buffer],
                index_buffer: render_data.index_buffer,
                images: vec![],
            };

            ctx.apply_pipeline(&pipelines[render_data.pipeline_handle as usize]);
            ctx.apply_bindings(&bindings);
            let screen_size = Vec2::new(WORLD_WIDTH, WORLD_HEIGHT);
            ctx.apply_uniforms(UniformsSource::table(&(mvp, shake_offset, screen_size)));
            ctx.draw(0, render_data.indices.len() as i32, 1);

            // Uncomment to draw hitboxes
            // /*
            // if !is_player {
            //     let num_segments = 32;
            //     let hitbox_vertices: Vec<Vec2> = (0..num_segments)
            //         .map(|i| {
            //             let angle = ((i as f32) * 2.0 * std::f32::consts::PI) / num_segments as f32;
            //             Vec2::new(angle.cos(), angle.sin())
            //         })
            //         .collect();

            //     let hitbox_indices: Vec<u16> = (1..num_segments-1)
            //         .flat_map(|i| vec![0, i as u16, (i+1) as u16])
            //         .collect();

            //     let hitbox_render_data = RenderDataCreator::create_render_data(ctx, &hitbox_vertices, &hitbox_indices, 0);
            //     let hitbox_model = Mat4::from_scale_rotation_translation(
            //         vec3(radius, radius, 1.0),
            //         rotation_quat,
            //         vec3(pos.x, pos.y, 0.0)
            //     );
            //     let hitbox_mvp = proj * hitbox_model;
            //     let hitbox_bindings = Bindings {
            //         vertex_buffers: vec![hitbox_render_data.vertex_buffer],
            //         index_buffer: hitbox_render_data.index_buffer,
            //         images: vec![],
            //     };

            //     ctx.apply_pipeline(&pipelines[0]);
            //     ctx.apply_bindings(&hitbox_bindings);
            //     ctx.apply_uniforms(UniformsSource::table(&hitbox_mvp));
            //     ctx.draw(0, hitbox_indices.len() as i32, 1);
            // }
            // */
        }
    }
}

struct AsteroidSpawner {
    spawn_timer: f32,
    spawn_interval: f32,
    difficulty_multiplier: f32,
}

impl AsteroidSpawner {
    fn default() -> Self {
        Self {
            spawn_timer: 0.0,
            spawn_interval: 2.0,
            difficulty_multiplier: 1.0,
        }
    }

    fn update(&mut self, world: &mut World, ctx: &mut Context, player_pos: Vec2) {
        self.spawn_timer += PHYSICS_FRAME_TIME;
        self.difficulty_multiplier += PHYSICS_FRAME_TIME * 0.01; // Increase difficulty over time

        if self.spawn_timer * self.difficulty_multiplier >= self.spawn_interval {
            self.spawn_timer = 0.0;

            let mut rng = rand::thread_rng();

            let mut pos = Vec2::new(
                rng.gen_range(0.0..WORLD_WIDTH),
                rng.gen_range(0.0..WORLD_HEIGHT)
            );
            while pos.distance(player_pos) < 100.0 {
                // TODO!
                pos = Vec2::new(rng.gen_range(0.0..WORLD_WIDTH), rng.gen_range(0.0..WORLD_HEIGHT));
            }

            let vel = (player_pos - pos).normalize() * ASTEROID_BASE_SPEED;
            let rot = rng.gen_range(0.0..std::f32::consts::PI * 2.0);

            world.add_asteroid(&mut *ctx, pos, vel, rot, rng.gen_range(1..4));
        }
    }
}
struct BackgroundRenderer {
    pipeline: Pipeline,
    quad_vertex_buffer: BufferId,
    quad_index_buffer: BufferId,
}
// Game Stage
struct Stage {
    world: World,
    ctx: Box<dyn RenderingBackend>,
    physics_elapsed_time: f32,
    physics_last_time: f64,
    draw_last_time: f64,
    pressed_keys: HashSet<KeyCode>,
    asteroid_spawner: AsteroidSpawner,
    background_renderer: BackgroundRenderer,
}

impl Stage {
    fn new() -> Self {
        let mut ctx = window::new_rendering_backend();
        let pipelines = Stage::load_pipelines(&mut *ctx);
        let mut world = World::new(&mut *ctx, pipelines);
        world.add_asteroid(
            &mut *ctx,
            Vec2::new(WORLD_WIDTH / 4.0, WORLD_HEIGHT / 4.0),
            Vec2::new(ASTEROID_BASE_SPEED, ASTEROID_BASE_SPEED),
            0.0,
            3
        );
        // for i in 1..5 {
        //     world.add_asteroid(
        //         &mut *ctx,
        //         Vec2::new(
        //             rand::thread_rng().gen_range(0.0..WORLD_WIDTH),
        //             rand::thread_rng().gen_range(0.0..WORLD_HEIGHT)
        //         ),
        //         Vec2::new(
        //             rand::thread_rng().gen_range(-1.0..1.0) * ASTEROID_BASE_SPEED,
        //             rand::thread_rng().gen_range(-1.0..1.0) * ASTEROID_BASE_SPEED
        //         ),
        //         rand::thread_rng().gen_range(0.0..std::f32::consts::PI * 2.0),
        //         rand::thread_rng().gen_range(1..4)
        //     );
        // }
        let background_shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: "#version 100          
                in vec2 pos;       
                void main() {
                    gl_Position = vec4(pos, 0.0, 1.0);

                }
            ",
                    fragment: include_str!("background_fragment.glsl"),
                },
                ShaderMeta {
                    uniforms: UniformBlockLayout {
                        uniforms: vec![],
                    },
                    images: vec![],
                }
            )
            .expect("Failed to load background shader.");
        let background_pipeline = ctx.new_pipeline(
            &[
                BufferLayout {
                    stride: 8,
                    ..Default::default()
                },
            ],
            &[VertexAttribute::new("pos", VertexFormat::Float2)],
            background_shader,
            PipelineParams::default()
        );
        let vertices_buffer = [
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        let quad_vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&vertices_buffer)
        );
        let quad_index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&[0, 1, 2, 0, 2, 3])
        );
        Stage {
            world,
            ctx,
            physics_elapsed_time: 0.0,
            physics_last_time: date::now(),
            draw_last_time: date::now(),
            pressed_keys: HashSet::new(),
            asteroid_spawner: AsteroidSpawner::default(),
            background_renderer: BackgroundRenderer {
                pipeline: background_pipeline,
                quad_vertex_buffer,
                quad_index_buffer,
            },
        }
    }
    fn load_pipelines(ctx: &mut Context) -> Vec<Pipeline> {
        let asteroid_shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: include_str!("vertex.glsl"),
                    fragment: include_str!("asteroid_fragment.glsl"),
                },
                ShaderMeta {
                    uniforms: UniformBlockLayout {
                        uniforms: vec![
                            UniformDesc::new("mvp", UniformType::Mat4),
                            UniformDesc::new("shake_offset", UniformType::Float2),
                            UniformDesc::new("screen_size", UniformType::Float2)
                        ],
                    },
                    images: vec![],
                }
            )
            .expect("Failed to load asteroid shader.");
        let default_shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: include_str!("vertex.glsl"),
                    fragment: include_str!("fragment.glsl"),
                },
                ShaderMeta {
                    uniforms: UniformBlockLayout {
                        uniforms: vec![
                            UniformDesc::new("mvp", UniformType::Mat4),
                            UniformDesc::new("shake_offset", UniformType::Float2),
                            UniformDesc::new("screen_size", UniformType::Float2)
                        ],
                    },
                    images: vec![],
                }
            )
            .expect("Failed to load default shader.");

        return vec![
            ctx.new_pipeline(
                &[
                    BufferLayout {
                        stride: 24,
                        ..Default::default()
                    },
                ],
                &[
                    VertexAttribute::new("pos", VertexFormat::Float2),
                    VertexAttribute::new("color0", VertexFormat::Float4),
                ],
                default_shader,
                PipelineParams::default()
            ),
            ctx.new_pipeline(
                &[
                    BufferLayout {
                        stride: 24,
                        ..Default::default()
                    },
                ],
                &[
                    VertexAttribute::new("pos", VertexFormat::Float2),
                    VertexAttribute::new("color0", VertexFormat::Float4),
                ],
                asteroid_shader,
                PipelineParams {
                    primitive_type: miniquad::PrimitiveType::Lines,
                    ..Default::default()
                }
            )
        ];
    }
    fn calculate_velocity(pressed_keys: &HashSet<KeyCode>) -> Vec2 {
        let mut velocity = Vec2::new(0.0, 0.0);
        if pressed_keys.contains(&KeyCode::W) {
            velocity.y = -1.0;
        }
        if pressed_keys.contains(&KeyCode::S) {
            velocity.y = 1.0;
        }
        if pressed_keys.contains(&KeyCode::A) {
            velocity.x = -1.0;
        }
        if pressed_keys.contains(&KeyCode::D) {
            velocity.x = 1.0;
        }
        if velocity.x != 0.0 && velocity.y != 0.0 {
            return velocity.normalize();
        }
        velocity
    }
    fn draw_background(&mut self) {
        self.ctx.apply_pipeline(&self.background_renderer.pipeline);
        self.ctx.apply_bindings(
            &(Bindings {
                vertex_buffers: vec![self.background_renderer.quad_vertex_buffer],
                index_buffer: self.background_renderer.quad_index_buffer,
                images: vec![],
            })
        );
        self.ctx.draw(0, 6, 1);
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {
        self.physics_elapsed_time += (date::now() - self.physics_last_time) as f32;
        self.physics_last_time = date::now();

        while self.physics_elapsed_time >= PHYSICS_FRAME_TIME {
            // let start = date::now();
            self.world.update(&mut *self.ctx);
            // let end = date::now();
            // println!("Physics update took: {}", end - start);
            // println!("FPS: {}", 1.0 / (end - start));
            // println!("Asteroids: {}", self.world.asteroids.positions.len());
            self.physics_elapsed_time -= PHYSICS_FRAME_TIME;
            let player_pos = self.world.player.pos;
            self.asteroid_spawner.update(&mut self.world, &mut *self.ctx, player_pos);
        }
    }

    fn draw(&mut self) {
        self.draw_background();
        let dt = (date::now() - self.draw_last_time) as f32;
        self.draw_last_time = date::now();
        match self.ctx.info().backend {
            Backend::OpenGl => {
                assert!(
                    self.world.asteroids.positions.len() == self.world.asteroids.rotations.len()
                );
                assert!(
                    self.world.asteroids.positions.len() == self.world.asteroids.collision.len()
                );
                assert!(
                    self.world.asteroids.positions.len() == self.world.asteroids.render_data.len()
                );
                assert!(self.world.bullets.positions.len() == self.world.bullets.rotations.len());
                assert!(self.world.bullets.positions.len() == self.world.bullets.collision.len());
                assert!(self.world.bullets.positions.len() == self.world.bullets.render_data.len());

                RenderSystem::draw(
                    &mut *self.ctx,
                    &self.world.asteroids.positions,
                    &self.world.asteroids.rotations,
                    &self.world.asteroids.collision
                        .iter()
                        .map(|c| c.radius)
                        .collect::<Vec<_>>(),
                    &self.world.asteroids.render_data,
                    &vec![EntityType::Asteroid; self.world.asteroids.positions.len()],
                    &self.world.pipelines,
                    &mut self.world.camera_shake,
                    dt
                );
                RenderSystem::draw(
                    &mut *self.ctx,
                    &self.world.bullets.positions,
                    &self.world.bullets.rotations,
                    &self.world.bullets.collision
                        .iter()
                        .map(|c| c.radius)
                        .collect::<Vec<_>>(),
                    &self.world.bullets.render_data,
                    &vec![EntityType::Bullet; self.world.bullets.positions.len()],
                    &self.world.pipelines,
                    &mut self.world.camera_shake,
                    dt
                );
                RenderSystem::draw(
                    &mut *self.ctx,
                    &vec![self.world.player.pos],
                    &vec![self.world.player.rot],
                    &vec![self.world.player.collision.radius],
                    &vec![self.world.player.render_data.clone()],
                    &vec![EntityType::Player],
                    &self.world.pipelines,
                    &mut self.world.camera_shake,
                    dt
                );
            }
            _ => {}
        }

        self.ctx.commit_frame();
    }
    fn key_down_event(&mut self, keycode: KeyCode, _keymods: KeyMods, _repeat: bool) {
        self.pressed_keys.insert(keycode);
        self.world.player.vel = Self::calculate_velocity(&self.pressed_keys) * 100.0;
        if keycode == KeyCode::Space {
            if self.world.player.is_dashing == 0 {
                self.world.player.is_dashing = DEFAULT_DASH_FRAMES;
            }
        }
    }

    fn key_up_event(&mut self, keycode: KeyCode, _keymods: KeyMods) {
        self.pressed_keys.remove(&keycode);
        self.world.player.vel = Self::calculate_velocity(&self.pressed_keys) * 100.0;
    }

    fn mouse_motion_event(&mut self, x: f32, y: f32) {
        let rot = (y - self.world.player.pos.y).atan2(x - self.world.player.pos.x);
        self.world.player.rot = rot + std::f32::consts::PI / 2.0;
    }
    fn mouse_button_down_event(&mut self, _button: MouseButton, _x: f32, _y: f32) {
        let pos = self.world.player.pos;
        let rot = self.world.player.rot;
        let bullet_rot = rot - std::f32::consts::PI / 2.0;
        let bullet_vel = Vec2::new(bullet_rot.cos(), bullet_rot.sin()) * 200.0;
        self.world.add_bullet(&mut *self.ctx, pos, bullet_vel, rot);
    }
}

fn main() {
    miniquad::start(
        Conf {
            window_title: "Rasteroids".to_owned(),
            window_width: WORLD_WIDTH as i32,
            window_height: WORLD_HEIGHT as i32,
            ..Default::default()
        },
        || Box::new(Stage::new())
    );
}
