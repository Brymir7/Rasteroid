use conf::Conf;
use miniquad::*;
use glam::{ vec3, Mat4, Quat, Vec2 };
use std::collections::{ HashMap, HashSet };

const WORLD_WIDTH: f32 = 800.0;
const WORLD_HEIGHT: f32 = 600.0;
const PHYSICS_FRAME_TIME: f32 = 1.0 / 60.0;
const PLAYER_HITBOX_SIZE: f32 = 20.0;
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
    components: Vec<Vec<PossibleComponent>>,
    pipelines: Vec<Pipeline>,
}

impl World {
    fn new(pipelines: Vec<Pipeline>) -> Self {
        World {
            entities: Vec::new(),
            next_entity_id: 0,
            components: vec![Vec::new(); PossibleComponent::VARIANT_COUNT],
            pipelines: pipelines,
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
            PossibleComponent::RenderData(RenderSystem::create_triangle_render_data(&mut *ctx, 0))
        );
    }
    fn add_asteroid(&mut self, ctx: &mut Context) {
        let asteroid = self.create_entity();
        self.add_component(
            asteroid,
            PossibleComponent::Position(Position(Vec2::new(WORLD_WIDTH / 2.0, WORLD_HEIGHT / 2.0)))
        );
        self.add_component(asteroid, PossibleComponent::Rotation(Rotation(0.0)));
        self.add_component(
            asteroid,
            PossibleComponent::Velocity(
                Velocity(Vec2::new(ASTEROID_BASE_SPEED, ASTEROID_BASE_SPEED))
            )
        );
        self.add_component(asteroid, PossibleComponent::Health(Health(100.0)));
        self.add_component(asteroid, PossibleComponent::Collision(Collision { radius: 20.0 }));
        self.add_component(asteroid, PossibleComponent::EntityType(EntityType::Asteroid));
        self.add_component(
            asteroid,
            PossibleComponent::RenderData(RenderSystem::create_asteroid_render_data(&mut *ctx, 1))
        );
    }
    fn add_bullet(&mut self, ctx: &mut Context, pos: Vec2, vel: Vec2, rot: f32) {
        let bullet = self.create_entity();
        self.add_component(bullet, PossibleComponent::Position(Position(pos)));
        self.add_component(bullet, PossibleComponent::Velocity(Velocity(vel)));
        self.add_component(bullet, PossibleComponent::Rotation(Rotation(rot)));
        self.add_component(bullet, PossibleComponent::Collision(Collision { radius: 5.0 }));
        self.add_component(bullet, PossibleComponent::EntityType(EntityType::Bullet));
        self.add_component(
            bullet,
            PossibleComponent::RenderData(RenderSystem::create_bullet_render_data(&mut *ctx, 0))
        );
    }
    fn update(&mut self) {
        MovementSystem::update(self);
        CollisionSystem::update(self);
    }
    fn draw(&self, ctx: &mut Context) {
        RenderSystem::update(self, ctx);
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

struct CollisionSystem;

impl CollisionSystem {
    fn update(world: &mut World) {
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
                        // Handle collision
                        CollisionSystem::handle_collision(world, entity1, entity2);
                    }
                }
            }
        }
    }

    fn handle_collision(world: &mut World, entity1: EntityId, entity2: EntityId) {
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

                if let Some(health) = world.get_health_component_mut(entity2) {
                    health.0 -= 10.0;
                    if health.0 <= 0.0 {
                        // TODO: Remove asteroid
                        println!("Asteroid destroyed!");
                    }
                }
                // TODO: Remove bullet
            }
            | (EntityType::Player, EntityType::Asteroid)
            | (EntityType::Asteroid, EntityType::Player) => {
                // Damage player
                if let Some(health) = world.get_health_component_mut(entity1) {
                    health.0 -= 10.0;
                    println!("Player hit! Health: {}", health.0);
                }
            }
            _ => {}
        }
    }
}

struct RenderSystem {}

impl RenderSystem {
    fn update(world: &World, ctx: &mut Context) {
        ctx.clear(Some((0.1, 0.2, 0.3, 1.0)), None, None);

        for &entity in &world.entities {
            if
                let (Some(pos), Some(render_data)) = (
                    world.get_position_component(entity),
                    world.get_render_data_component(entity),
                )
            {
                let mut rotation = Quat::IDENTITY;
                if let Some(entity_rot) = world.get_rotation_component(entity) {
                    rotation = Quat::from_rotation_z(entity_rot.0);
                }
                let model = Mat4::from_scale_rotation_translation(
                    vec3(DEFAULT_SCALE_FACTOR, DEFAULT_SCALE_FACTOR, 1.0),
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
            }
        }
    }
}

// Game Stage
struct Stage {
    world: World,
    ctx: Box<dyn RenderingBackend>,
    elapsed_time: f32,
    last_time: f64,
    pressed_keys: HashSet<KeyCode>,
}

impl Stage {
    fn new() -> Self {
        let mut ctx = window::new_rendering_backend();
        let pipelines = Stage::load_pipelines(&mut *ctx);
        let mut world = World::new(pipelines);
        world.add_player(&mut *ctx);
        world.add_asteroid(&mut *ctx);

        Stage {
            world,
            ctx,
            elapsed_time: 0.0,
            last_time: date::now(),
            pressed_keys: HashSet::new(),
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
            return  velocity.normalize()
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
        }
    }

    fn draw(&mut self) {
        self.ctx.clear(Some((1.0, 1.0, 1.0, 1.0)), None, None);
        match self.ctx.info().backend {
            Backend::OpenGl => {
                self.world.draw(&mut *self.ctx);
            }
            _ => {}
        }
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

impl RenderSystem {
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
            Vec2::new(-0.1, 0.3),
            Vec2::new(0.1, 0.3),
            Vec2::new(0.1, -0.3),
            Vec2::new(-0.1, -0.3)
        ];
        Self::create_render_data(ctx, &vertices, &[0, 1, 2, 0, 2, 3], pipeline_handle)
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
            window_title: "Asteroids ECS".to_owned(),
            window_width: WORLD_WIDTH as i32,
            window_height: WORLD_HEIGHT as i32,
            ..Default::default()
        },
        || Box::new(Stage::new())
    );
}
