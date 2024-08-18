use conf::Conf;
use miniquad::*;
use glam::{ vec3, Mat4, Quat, Vec2 };
use rand::Rng;
use std::collections::HashSet;

const WORLD_WIDTH: f32 = 800.0;
const WORLD_HEIGHT: f32 = 600.0;
const PHYSICS_FRAME_TIME: f32 = 1.0 / 60.0;
const PLAYER_HITBOX_SIZE: f32 = 10.0;
const ASTEROID_BASE_SPEED: f32 = 50.0;
const DEFAULT_SCALE_FACTOR: f32 = 20.0;
const ENTITY_ID_PLAYER: EntityId = 0;
type EntityId = usize;
#[derive(Clone)]
enum PossibleComponent {
    Position(Position),
    Rotation(Rotation),
    Velocity(Velocity),
    Health(Health),
    Collision(Collision),
    EntityType(EntityType),
    RenderData(RenderData),
}
impl PossibleComponent {
    const VARIANT_COUNT: usize = 7;
    fn component_to_index(&self) -> usize {
        match self {
            PossibleComponent::Position(_) => 0,
            PossibleComponent::Rotation(_) => 1,
            PossibleComponent::Velocity(_) => 2,
            PossibleComponent::Health(_) => 3,
            PossibleComponent::Collision(_) => 4,
            PossibleComponent::EntityType(_) => 5,
            PossibleComponent::RenderData(_) => 6,
        }
    }
}
struct World {
    entities: Vec<EntityId>,
    next_entity_id: EntityId,
    components: Vec<Vec<PossibleComponent>>, // refactor into multiple members of World
    pipelines: Vec<Pipeline>, // refactor into static size array, when all pipelines are known
    available_entity_ids: Vec<EntityId>,
}

impl World {
    fn new(pipelines: Vec<Pipeline>) -> Self {
        World {
            entities: Vec::new(),
            next_entity_id: 0,
            components: vec![Vec::new(); PossibleComponent::VARIANT_COUNT],
            pipelines: pipelines,
            available_entity_ids: Vec::new(),
        }
    }
    fn add_component(&mut self, entity: EntityId, component: PossibleComponent) {
        let index = component.component_to_index();
        assert!(index < PossibleComponent::VARIANT_COUNT);
        if self.components[index].len() <= entity {
            self.components[index].resize(entity + 1, component);
        } else {
            self.components[index][entity] = component;
        }
    }

    fn get_position_component(&self, entity: EntityId) -> Option<&Position> {
        self.components[0].get(entity).and_then(|c| {
            match c {
                PossibleComponent::Position(p) => Some(p),
                _ => None,
            }
        })
    }
    fn get_position_component_mut(&mut self, entity: EntityId) -> Option<&mut Position> {
        self.components[0].get_mut(entity).and_then(|c| {
            match c {
                PossibleComponent::Position(p) => Some(p),
                _ => None,
            }
        })
    }
    fn get_rotation_component(&self, entity: EntityId) -> Option<&Rotation> {
        self.components[1].get(entity).and_then(|c| {
            match c {
                PossibleComponent::Rotation(r) => Some(r),
                _ => None,
            }
        })
    }
    fn get_rotation_component_mut(&mut self, entity: EntityId) -> Option<&mut Rotation> {
        self.components[1].get_mut(entity).and_then(|c| {
            match c {
                PossibleComponent::Rotation(r) => Some(r),
                _ => None,
            }
        })
    }
    fn get_velocity_component(&self, entity: EntityId) -> Option<&Velocity> {
        self.components[2].get(entity).and_then(|c| {
            match c {
                PossibleComponent::Velocity(v) => Some(v),
                _ => None,
            }
        })
    }
    fn get_velocity_component_mut(&mut self, entity: EntityId) -> Option<&mut Velocity> {
        self.components[2].get_mut(entity).and_then(|c| {
            match c {
                PossibleComponent::Velocity(v) => Some(v),
                _ => None,
            }
        })
    }
    fn get_health_component(&self, entity: EntityId) -> Option<&Health> {
        self.components[3].get(entity).and_then(|c| {
            match c {
                PossibleComponent::Health(h) => Some(h),
                _ => None,
            }
        })
    }
    fn get_health_component_mut(&mut self, entity: EntityId) -> Option<&mut Health> {
        self.components[3].get_mut(entity).and_then(|c| {
            match c {
                PossibleComponent::Health(h) => Some(h),
                _ => None,
            }
        })
    }
    fn get_collision_component(&self, entity: EntityId) -> Option<&Collision> {
        self.components[4].get(entity).and_then(|c| {
            match c {
                PossibleComponent::Collision(c) => Some(c),
                _ => None,
            }
        })
    }
    fn get_collision_component_mut(&mut self, entity: EntityId) -> Option<&mut Collision> {
        self.components[4].get_mut(entity).and_then(|c| {
            match c {
                PossibleComponent::Collision(c) => Some(c),
                _ => None,
            }
        })
    }
    fn get_entity_type_component(&self, entity: EntityId) -> Option<&EntityType> {
        self.components[5].get(entity).and_then(|c| {
            match c {
                PossibleComponent::EntityType(t) => Some(t),
                _ => None,
            }
        })
    }

    fn get_render_data_component(&self, entity: EntityId) -> Option<&RenderData> {
        self.components[6].get(entity).and_then(|c| {
            match c {
                PossibleComponent::RenderData(r) => Some(r),
                _ => None,
            }
        })
    }

    fn get_mut_pos_and_vel(&mut self, entity: EntityId) -> Option<(&mut Position, &mut Velocity)> {
        let (pos, vel_and_rest) = self.components.split_at_mut(1);
        let (_, vel) = vel_and_rest.split_at_mut(1);
        if let Some(pos) = pos[0].get_mut(entity) {
            if let Some(vel) = vel[0].get_mut(entity) {
                match (pos, vel) {
                    (PossibleComponent::Position(pos), PossibleComponent::Velocity(vel)) => {
                        return Some((pos, vel));
                    }
                    _ => {}
                }
            }
        }
        None
    }
    fn get_mut_pos_and_rot(&mut self, entity: EntityId) -> Option<(&mut Position, &mut Rotation)> {
        let (pos, rot_and_rest) = self.components.split_at_mut(1);
        let (rot, _) = rot_and_rest.split_at_mut(1);
        if let Some(pos) = pos[0].get_mut(entity) {
            if let Some(rot) = rot[0].get_mut(entity) {
                match (pos, rot) {
                    (PossibleComponent::Position(pos), PossibleComponent::Rotation(rot)) => {
                        return Some((pos, rot));
                    }
                    _ => {}
                }
            }
        }
        None
    }
    fn create_entity(&mut self) -> EntityId {
        // if let Some(entity) = self.available_entity_ids.pop() {
        //     return entity;
        // }
        let entity = self.next_entity_id;
        self.entities.push(entity);
        self.next_entity_id += 1;
        entity
    }
    fn add_player(&mut self, ctx: &mut Context) {
        let player = self.create_entity();
        self.add_component(
            player,
            PossibleComponent::Position(Position(Vec2::new(WORLD_WIDTH / 2.0, WORLD_HEIGHT / 2.0)))
        );
        self.add_component(player, PossibleComponent::Rotation(Rotation(0.0)));
        self.add_component(player, PossibleComponent::Velocity(Velocity(Vec2::new(0.0, 0.0))));
        self.add_component(player, PossibleComponent::Health(Health(100.0)));
        self.add_component(
            player,
            PossibleComponent::Collision(Collision { radius: PLAYER_HITBOX_SIZE })
        );
        self.add_component(player, PossibleComponent::EntityType(EntityType::Player));
        self.add_component(
            player,
            PossibleComponent::RenderData(
                RenderDataCreator::create_triangle_render_data(&mut *ctx, 0)
            )
        );
    }
    fn add_asteroid(&mut self, ctx: &mut Context, pos: Vec2, vel: Vec2, rot: f32) {
        let asteroid = self.create_entity();
        self.add_component(asteroid, PossibleComponent::Position(Position(pos)));
        self.add_component(asteroid, PossibleComponent::Velocity(Velocity(vel)));
        self.add_component(asteroid, PossibleComponent::Rotation(Rotation(rot)));
        self.add_component(asteroid, PossibleComponent::Health(Health(10.0)));
        self.add_component(asteroid, PossibleComponent::Collision(Collision { radius: 20.0 }));
        self.add_component(asteroid, PossibleComponent::EntityType(EntityType::Asteroid));
        self.add_component(
            asteroid,
            PossibleComponent::RenderData(
                RenderDataCreator::create_asteroid_render_data(&mut *ctx, 1)
            )
        );
    }
    fn add_bullet(&mut self, ctx: &mut Context, pos: Vec2, vel: Vec2, rot: f32) {
        let bullet = self.create_entity();
        self.add_component(bullet, PossibleComponent::Position(Position(pos)));
        self.add_component(bullet, PossibleComponent::Velocity(Velocity(vel)));
        self.add_component(bullet, PossibleComponent::Rotation(Rotation(rot)));
        self.add_component(bullet, PossibleComponent::Health(Health(1.0)));
        self.add_component(bullet, PossibleComponent::Collision(Collision { radius: 5.0 }));
        self.add_component(bullet, PossibleComponent::EntityType(EntityType::Bullet));
        self.add_component(
            bullet,
            PossibleComponent::RenderData(
                RenderDataCreator::create_bullet_render_data(&mut *ctx, 0)
            )
        );
    }
    fn remove_entity(&mut self, entity: EntityId) {
        self.entities[entity] = std::usize::MAX;
        self.available_entity_ids.push(entity);
    }
    fn handle_collision_events(&mut self, events: Vec<CollisionEvent>) {
        for event in events {
            match event.event_type {
                CollisionEventType::PlayerHit => {
                    self.remove_entity(event.triggered_by);
                }
                CollisionEventType::PlayerDeath => {
                    self.remove_entity(event.target);
                }
                CollisionEventType::AsteroidKilled => {
                    self.remove_entity(event.target);
                }
                CollisionEventType::BulletDestroyed => {
                    self.remove_entity(event.target);
                }
            }
        }
    }

    fn update(&mut self) {
        MovementSystem::update(self);
        let collision_events = CollisionSystem::update(self);
        self.handle_collision_events(collision_events);
    }
    fn draw(&self, ctx: &mut Context) {
        RenderSystem::draw(self, ctx);
    }
}

// Components
#[derive(Clone, Copy)]
struct Position(Vec2);
#[derive(Clone, Copy)]
struct Rotation(f32);
#[derive(Clone, Copy)]
struct Velocity(Vec2);

#[derive(Clone, Copy)]
struct Health(f32);

#[derive(Clone, Copy)]
struct Collision {
    radius: f32,
}

#[derive(Clone)]
struct RenderData {
    vertices: Vec<Vec2>,
    vertex_buffer: BufferId,
    index_buffer: BufferId,
    pipeline_handle: u8,
}
#[derive(Clone, Copy)]
enum EntityType {
    Player,
    Asteroid,
    Bullet,
}

// Systems
struct MovementSystem;
impl MovementSystem {
    fn update(world: &mut World) {
        let entities_to_update: Vec<EntityId> = world.entities.iter().cloned().collect(); // NOTE avoid borrowing issues
        for entity in entities_to_update {
            let mut_pos_vel = world.get_mut_pos_and_vel(entity);
            if let Some(mut_pos_vel) = mut_pos_vel {
                let (pos, vel) = mut_pos_vel;
                pos.0 += vel.0 * PHYSICS_FRAME_TIME;
            }
        }
    }
}
enum CollisionEventType {
    PlayerHit,
    PlayerDeath,
    AsteroidKilled,
    BulletDestroyed,
}
struct CollisionEvent {
    event_type: CollisionEventType,
    triggered_by: EntityId,
    target: EntityId,
}
struct CollisionSystem;

impl CollisionSystem {
    fn update(world: &mut World) -> Vec<CollisionEvent> {
        let mut events: Vec<CollisionEvent> = Vec::new();
        let entities: Vec<EntityId> = world.entities.clone();
        for (i, &entity1) in entities.iter().enumerate() {
            for &entity2 in entities.iter().skip(i + 1) {
                if
                    let (Some(pos1), Some(col1), Some(pos2), Some(col2)) = (
                        world.get_position_component(entity1),
                        world.get_collision_component(entity1),
                        world.get_position_component(entity2),
                        world.get_collision_component(entity2),
                    )
                {
                    let distance = pos1.0.distance(pos2.0);
                    if distance < col1.radius + col2.radius {
                        CollisionSystem::handle_collision(world, entity1, entity2, &mut events);
                    }
                }
            }
        }
        events
    }

    fn handle_collision(
        world: &mut World,
        entity1: EntityId,
        entity2: EntityId,
        result: &mut Vec<CollisionEvent>
    ) {
        // Simple collision response: destroy bullets and damage asteroids/player
        let type1 = world.get_entity_type_component(entity1).expect("Entity missing type");
        let type2 = world.get_entity_type_component(entity2).expect("Entity missing type");

        match (type1, type2) {
            | (EntityType::Bullet, EntityType::Asteroid)
            | (EntityType::Asteroid, EntityType::Bullet) => {
                let (bullet, asteroid) = if let EntityType::Bullet = type1 {
                    (entity1, entity2)
                } else {
                    (entity2, entity1)
                };

                if let Some(health) = world.get_health_component_mut(asteroid) {
                    health.0 -= 10.0;
                    if health.0 <= 0.0 {
                        result.push(CollisionEvent {
                            event_type: CollisionEventType::AsteroidKilled,
                            triggered_by: bullet,
                            target: asteroid,
                        });
                    }
                }
                if let Some(health) = world.get_health_component_mut(bullet) {
                    health.0 -= 1.0;
                    if health.0 <= 0.0 {
                        result.push(CollisionEvent {
                            event_type: CollisionEventType::BulletDestroyed,
                            triggered_by: asteroid,
                            target: bullet,
                        });
                    }
                }
            }
            | (EntityType::Player, EntityType::Asteroid)
            | (EntityType::Asteroid, EntityType::Player) => {
                let (player, asteroid) = if let EntityType::Player = type1 {
                    (entity1, entity2)
                } else {
                    (entity2, entity1)
                };
                if let Some(health) = world.get_health_component_mut(entity1) {
                    health.0 -= 10.0;
                    println!("Player hit! Health: {}", health.0);
                    if health.0 <= 0.0 {
                        return result.push(CollisionEvent {
                            event_type: CollisionEventType::PlayerDeath,
                            triggered_by: asteroid,
                            target: player,
                        });
                    }
                    result.push(CollisionEvent {
                        event_type: CollisionEventType::PlayerHit,
                        triggered_by: asteroid,
                        target: player,
                    });
                }
            }
            _ => {}
        }
    }
}

struct RenderSystem {}

impl RenderSystem {
    fn draw(world: &World, ctx: &mut Context) {
        for &entity in &world.entities {
            let is_player = entity == ENTITY_ID_PLAYER;
            if
                let (Some(pos), Some(render_data), Some(radius)) = (
                    world.get_position_component(entity),
                    world.get_render_data_component(entity),
                    world.get_collision_component(entity),
                )
            {
                let mut rotation = Quat::IDENTITY;
                if let Some(entity_rot) = world.get_rotation_component(entity) {
                    rotation = Quat::from_rotation_z(entity_rot.0);
                }
                let scale = if is_player {
                    vec3(PLAYER_HITBOX_SIZE * 1.5, PLAYER_HITBOX_SIZE * 1.5, 1.0) // draw bigger than hitbox
                } else {
                    vec3(radius.radius * 0.8, radius.radius * 0.8, 1.0)
                };
                let model = Mat4::from_scale_rotation_translation(
                    scale,
                    rotation,
                    vec3(pos.0.x, pos.0.y, 0.0)
                );
                let proj = Mat4::orthographic_rh_gl(0.0, WORLD_WIDTH, WORLD_HEIGHT, 0.0, -1.0, 1.0);
                let mvp = proj * model;

                let bindings = Bindings {
                    vertex_buffers: vec![render_data.vertex_buffer],
                    index_buffer: render_data.index_buffer,
                    images: vec![],
                };

                ctx.apply_pipeline(&world.pipelines[render_data.pipeline_handle as usize]);
                ctx.apply_bindings(&bindings);
                ctx.apply_uniforms(UniformsSource::table(&mvp));
                ctx.draw(0, render_data.vertices.len() as i32, 1);
                // draw hitboxes
                // if !is_player {
                //     let num_segments = 32; // Increase for a smoother circle
                //     let hitbox_vertices = (0..num_segments)
                //         .map(|i| {
                //             let angle = ((i as f32) * 2.0 * std::f32::consts::PI) / num_segments as f32;
                //             Vec2::new(angle.cos(), angle.sin())
                //         })
                //         .collect::<Vec<_>>();

                //     let mut hitbox_indices = Vec::new();
                //     for i in 1..(num_segments - 1) {
                //         hitbox_indices.push(0);
                //         hitbox_indices.push(i as u16);
                //         hitbox_indices.push((i + 1) as u16);
                //     }

                //     let hitbox_render_data = RenderDataCreator::create_render_data(ctx, &hitbox_vertices, &hitbox_indices, 0);
                //     let hitbox_model = Mat4::from_scale_rotation_translation(
                //         vec3(radius.radius*0.5, radius.radius*0.5, 1.0),
                //         rotation,
                //         vec3(pos.0.x, pos.0.y, 0.0)
                //     );
                //     let hitbox_mvp = proj * hitbox_model;
                //     let hitbox_bindings = Bindings {
                //         vertex_buffers: vec![hitbox_render_data.vertex_buffer],
                //         index_buffer: hitbox_render_data.index_buffer,
                //         images: vec![],
                //     };

                //     ctx.apply_pipeline(&world.pipelines[0]);
                //     ctx.apply_bindings(&hitbox_bindings);

                //     ctx.apply_uniforms(UniformsSource::table(&hitbox_mvp));

                //     ctx.draw(0, hitbox_indices.len() as i32, 1);
            }
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

        if self.spawn_timer >= self.spawn_interval {
            self.spawn_timer = 0.0;

            let mut rng = rand::thread_rng();

            // Spawn at a safe distance from the player (e.g., 100 units away)
            let mut pos = Vec2::new(
                rng.gen_range(0.0..WORLD_WIDTH),
                rng.gen_range(0.0..WORLD_HEIGHT)
            );
            while pos.distance(player_pos) < 100.0 {
                pos = Vec2::new(rng.gen_range(0.0..WORLD_WIDTH), rng.gen_range(0.0..WORLD_HEIGHT));
            }

            let vel =
                (player_pos - pos).normalize() * ASTEROID_BASE_SPEED * self.difficulty_multiplier;
            let rot = rng.gen_range(0.0..std::f32::consts::PI * 2.0);

            world.add_asteroid(&mut *ctx, pos, vel, rot);
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
    elapsed_time: f32,
    last_time: f64,
    pressed_keys: HashSet<KeyCode>,
    asteroid_spawner: AsteroidSpawner,
    background_renderer: BackgroundRenderer,
}

impl Stage {
    fn new() -> Self {
        let mut ctx = window::new_rendering_backend();
        let pipelines = Stage::load_pipelines(&mut *ctx);
        let mut world = World::new(pipelines);
        world.add_player(&mut *ctx);
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
                        uniforms: vec![UniformDesc::new("u_time", UniformType::Float1)],
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
            elapsed_time: 0.0,
            last_time: date::now(),
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
                    vertex: include_str!("asteroid_vertex.glsl"),
                    fragment: include_str!("asteroid_fragment.glsl"),
                },
                ShaderMeta {
                    uniforms: UniformBlockLayout {
                        uniforms: vec![UniformDesc::new("mvp", UniformType::Mat4)],
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
                        uniforms: vec![UniformDesc::new("mvp", UniformType::Mat4)],
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
                PipelineParams::default()
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
}

impl EventHandler for Stage {
    fn update(&mut self) {
        self.elapsed_time += (date::now() - self.last_time) as f32;
        self.last_time = date::now();

        while self.elapsed_time >= PHYSICS_FRAME_TIME {
            self.world.update();
            self.elapsed_time -= PHYSICS_FRAME_TIME;
            let player_pos = self.world
                .get_position_component(ENTITY_ID_PLAYER)
                .expect("Player Entity not at 0").0;
            self.asteroid_spawner.update(&mut self.world, &mut *self.ctx, player_pos);
        }
    }
    fn draw(&mut self) {
        self.ctx.clear(Some((0.2, 0.2, 0.6, 1.0)), None, None); // Clear to black

        self.ctx.apply_pipeline(&self.background_renderer.pipeline);
        self.ctx.apply_bindings(
            &(Bindings {
                vertex_buffers: vec![self.background_renderer.quad_vertex_buffer],
                index_buffer: self.background_renderer.quad_index_buffer,
                images: vec![],
            })
        );
        // self.ctx.apply_uniforms(UniformsSource::table(&Vec2::new(self.frame as f32 / 25.0, 0.0)));

        self.ctx.draw(0, 6, 1);
        match self.ctx.info().backend {
            Backend::OpenGl => {
                self.world.draw(&mut *self.ctx);
            }
            _ => {}
        }

        self.ctx.commit_frame();
    }
    fn key_down_event(&mut self, keycode: KeyCode, _keymods: KeyMods, _repeat: bool) {
        assert!(match self.world.get_entity_type_component(ENTITY_ID_PLAYER).unwrap() {
            EntityType::Player => true,
            _ => false,
        });
        self.pressed_keys.insert(keycode);
        if let Some(vel) = self.world.get_velocity_component_mut(ENTITY_ID_PLAYER) {
            vel.0 = Self::calculate_velocity(&self.pressed_keys) * 100.0;
        }
    }

    fn key_up_event(&mut self, keycode: KeyCode, _keymods: KeyMods) {
        self.pressed_keys.remove(&keycode);
        if let Some(vel) = self.world.get_velocity_component_mut(ENTITY_ID_PLAYER) {
            vel.0 = Self::calculate_velocity(&self.pressed_keys) * 100.0;
        }
    }

    fn mouse_motion_event(&mut self, x: f32, y: f32) {
        assert!(match self.world.get_entity_type_component(ENTITY_ID_PLAYER).unwrap() {
            EntityType::Player => true,
            _ => false,
        });

        if let Some(rot_and_vel) = self.world.get_mut_pos_and_rot(ENTITY_ID_PLAYER) {
            let (pos, rot) = rot_and_vel;

            *rot = Rotation((y - pos.0.y).atan2(x - pos.0.x));
            *rot = Rotation(rot.0 + std::f32::consts::PI / 2.0);
        }
    }
    fn mouse_button_down_event(&mut self, _button: MouseButton, _x: f32, _y: f32) {
        assert!(match self.world.get_entity_type_component(ENTITY_ID_PLAYER).unwrap() {
            EntityType::Player => true,
            _ => false,
        });

        if let Some(pos) = self.world.get_position_component(ENTITY_ID_PLAYER) {
            let rot = self.world.get_rotation_component(ENTITY_ID_PLAYER).unwrap().0;
            let bullet_rot = rot - std::f32::consts::PI / 2.0;
            let bullet_vel = Vec2::new(bullet_rot.cos(), bullet_rot.sin()) * 200.0;
            self.world.add_bullet(&mut *self.ctx, pos.0, bullet_vel, rot);
        }
    }
}
struct RenderDataCreator;
impl RenderDataCreator {
    fn create_triangle_render_data(ctx: &mut Context, pipeline_handle: u8) -> RenderData {
        let vertices = vec![Vec2::new(-0.5, 0.25), Vec2::new(0.5, 0.25), Vec2::new(0.0, -1.0)];
        Self::create_render_data(ctx, &vertices, &[0, 1, 2], pipeline_handle)
    }

    fn create_asteroid_render_data(ctx: &mut Context, pipeline_handle: u8) -> RenderData {
        let vertices = (0..8)
            .map(|i| {
                let angle = ((i as f32) * std::f32::consts::PI) / 4.0;
                Vec2::new(angle.cos(), angle.sin())
            })
            .collect::<Vec<_>>();
        let indices = (1..7).flat_map(|i| vec![0, i, i + 1]).collect::<Vec<_>>();
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
            vertices: vertices.to_vec(),
            vertex_buffer,
            index_buffer,
            pipeline_handle,
        }
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
