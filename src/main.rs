use conf::Conf;
use miniquad::*;
use glam::{ vec3, Mat4, Quat, Vec2 };
use std::collections::HashMap;

const WORLD_WIDTH: f32 = 800.0;
const WORLD_HEIGHT: f32 = 600.0;
const PHYSICS_FRAME_TIME: f32 = 1.0 / 60.0;
const PLAYER_HITBOX_SIZE: f32 = 20.0;
const ASTEROID_BASE_SPEED: f32 = 50.0;
const DEFAULT_SCALE_FACTOR: f32 = 20.0;


type EntityId = usize;

struct World {
    entities: Vec<EntityId>,
    next_entity_id: EntityId,
    component_type_to_handle: HashMap<std::any::TypeId, u32>,
    handle_to_components: Vec<Box<dyn AnyMap>>,
    pipelines: Vec<Pipeline>,
}

trait AnyMap {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl<T: 'static> AnyMap for HashMap<EntityId, T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl World {
    fn new(pipelines: Vec<Pipeline>) -> Self {
        World {
            entities: Vec::new(),
            next_entity_id: 0,
            component_type_to_handle: HashMap::new(),
            handle_to_components: Vec::new(),
            pipelines: pipelines,
        }
    }
    fn get_component<T: 'static>(&self, entity: EntityId) -> Option<&T> {
        let type_id = std::any::TypeId::of::<T>();
        if let Some(handle) = self.component_type_to_handle.get(&type_id) {
            if let Some(map) = self.handle_to_components.get(*handle as usize) {
                if let Some(components) = map.as_any().downcast_ref::<HashMap<EntityId, T>>() {
                    return components.get(&entity);
                }
            }
        }
        None
    }
    fn get_component_mut<T: 'static>(&mut self, entity: EntityId) -> Option<&mut T> {
        let type_id = std::any::TypeId::of::<T>();
        if let Some(handle) = self.component_type_to_handle.get(&type_id) {
            if let Some(map) = self.handle_to_components.get_mut(*handle as usize) {
                if let Some(components) = map.as_any_mut().downcast_mut::<HashMap<EntityId, T>>() {
                    return components.get_mut(&entity);
                }
            }
        }
        None
    }
    fn add_component<T: 'static>(&mut self, entity: EntityId, component: T) {
        let type_id = std::any::TypeId::of::<T>();
        if let Some(handle) = self.component_type_to_handle.get(&type_id) {
            if let Some(map) = self.handle_to_components.get_mut(*handle as usize) {
                if let Some(components) = map.as_any_mut().downcast_mut::<HashMap<EntityId, T>>() {
                    components.insert(entity, component);
                    return;
                }
            }
        }
        let mut new_map = HashMap::new();
        new_map.insert(entity, component);
        let new_handle = self.handle_to_components.len() as u32;
        self.component_type_to_handle.insert(type_id, new_handle);
        self.handle_to_components.push(Box::new(new_map));
    }
    fn get_tuple_components_mut<T1: 'static, T2: 'static>(&mut self, entity: EntityId) -> Option<(&mut T1, &mut T2)> {
        let type_id1 = std::any::TypeId::of::<T1>();
        let type_id2 = std::any::TypeId::of::<T2>();
        assert!(type_id1 != type_id2); // u dont need to get the same type twice
        if let (Some(handle1), Some(handle2)) = (
            self.component_type_to_handle.get(&type_id1),
            self.component_type_to_handle.get(&type_id2)
        ) {
            let (left, right) = self.handle_to_components.split_at_mut(*handle2 as usize);
            if let Some(map1) = left.get_mut(*handle1 as usize) {
                if let Some(map2) = right.get_mut(0) {
                    if let Some(components1) = map1.as_any_mut().downcast_mut::<HashMap<EntityId, T1>>() {
                        if let Some(components2) = map2.as_any_mut().downcast_mut::<HashMap<EntityId, T2>>() { 
                            if let (Some(comp1), Some(comp2)) = (components1.get_mut(&entity), components2.get_mut(&entity)) {
                                return Some((comp1, comp2));
                            }
                        }
                    }
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
struct Velocity(Vec2);

#[derive(Clone, Copy)]
struct Health(f32);

#[derive(Clone, Copy)]
struct Collision {
    radius: f32,
}

struct RenderData {
    vertices: Vec<Vec2>,
    vertex_buffer: BufferId,
    index_buffer: BufferId,
    pipeline_handle: u8,
}

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

            let mut_pos_vel = world.get_tuple_components_mut::<Position, Velocity>(entity);
            if let Some(mut_pos_vel) = mut_pos_vel {
                let ( pos,  vel) = mut_pos_vel;
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
                        world.get_component::<Position>(entity1),
                        world.get_component::<Collision>(entity1),
                        world.get_component::<Position>(entity2),
                        world.get_component::<Collision>(entity2),
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
        let type1 = world.get_component::<EntityType>(entity1).unwrap();
        let type2 = world.get_component::<EntityType>(entity2).unwrap();

        match (type1, type2) {
            | (EntityType::Bullet, EntityType::Asteroid)
            | (EntityType::Asteroid, EntityType::Bullet) => {
                let (bullet, asteroid) = if let EntityType::Bullet = type1 {
                    (entity1, entity2)
                } else {
                    (entity2, entity1)
                };

                if let Some(health) = world.get_component_mut::<Health>(asteroid) {
                    health.0 -= 10.0;
                    if health.0 <= 0.0 {
                        // TODO: Remove asteroid
                    }
                }
                // TODO: Remove bullet
            }
            | (EntityType::Player, EntityType::Asteroid)
            | (EntityType::Asteroid, EntityType::Player) => {
                // Damage player
                if let Some(health) = world.get_component_mut::<Health>(entity1) {
                    health.0 -= 10.0;
                }
            }
            _ => {}
        }
    }
}

struct RenderSystem {

}

impl RenderSystem {
    fn update(world: &World, ctx: &mut Context) {
        ctx.clear(Some((0.1, 0.2, 0.3, 1.0)), None, None);

        for &entity in &world.entities {
            if
                let (Some(pos), Some(render_data)) = (
                    world.get_component::<Position>(entity),
                    world.get_component::<RenderData>(entity),
                )
            {
                let model = Mat4::from_scale_rotation_translation(
                    vec3(DEFAULT_SCALE_FACTOR, DEFAULT_SCALE_FACTOR, 1.0),
                    Quat::IDENTITY,
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
}

impl Stage {
    fn new() -> Self {
        let mut ctx = window::new_rendering_backend();
        let pipelines = Stage::load_shaders(&mut *ctx);
        let mut world = World::new(pipelines);
        // Create player
        let player = world.create_entity();
        world.add_component(player, Position(Vec2::new(WORLD_WIDTH / 2.0, WORLD_HEIGHT / 2.0)));
        world.add_component(player, Velocity(Vec2::ZERO));
        world.add_component(player, Health(100.0));
        world.add_component(player, Collision { radius: PLAYER_HITBOX_SIZE / 2.0 });
        world.add_component(player, EntityType::Player);
        world.add_component(
            player,
            RenderSystem::create_triangle_render_data(&mut *ctx, 0)
        );

        let asteroid = world.create_entity();
        world.add_component(asteroid, Position(Vec2::new(100.0, 100.0)));
        world.add_component(
            asteroid,
            Velocity(Vec2::new(ASTEROID_BASE_SPEED, ASTEROID_BASE_SPEED))
        );
        world.add_component(asteroid, Health(100.0));
        world.add_component(asteroid, Collision { radius: 20.0 });
        world.add_component(asteroid, EntityType::Asteroid);
        world.add_component(
            asteroid,
            RenderSystem::create_asteroid_render_data(&mut *ctx, 1)
        );

        Stage {
            world,
            ctx,
            elapsed_time: 0.0,
            last_time: date::now(),
        }
    }
    fn load_shaders(ctx: &mut Context) -> Vec<Pipeline> {
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
        assert!(match self.world.get_component::<EntityType>(0).unwrap() {
            EntityType::Player => true,
            _ => false,
        });
        if let Some(vel) = self.world.get_component_mut::<Velocity>(0) {
            // Assuming player is always entity 0
            println!("Key down: {:?}", keycode);
            println!("Player velocity: {:?}", vel.0);
            match keycode {
                KeyCode::W => {
                    vel.0.y -= 100.0;
                }
                KeyCode::S => {
                    vel.0.y += 100.0;
                }
                KeyCode::A => {
                    vel.0.x -= 100.0;
                }
                KeyCode::D => {
                    vel.0.x += 100.0;
                }
                KeyCode::Space => {
                    let bullet = self.world.create_entity();
                    if let Some(player_pos) = self.world.get_component::<Position>(0) {
                        self.world.add_component(bullet, Position(player_pos.0));
                        self.world.add_component(bullet, Velocity(Vec2::new(0.0, -200.0)));
                        self.world.add_component(bullet, Collision { radius: 5.0 });
                        self.world.add_component(bullet, EntityType::Bullet);
                        self.world.add_component(
                            bullet,
                            RenderSystem::create_bullet_render_data(
                                &mut *self.ctx,
                                0
                            )
                        );
                    }
                }
                _ => {}
            }
        }
    }
}

impl RenderSystem {
    fn create_triangle_render_data(ctx: &mut Context, pipeline_handle: u8) -> RenderData {
        let vertices = vec![Vec2::new(-0.5, 0.5), Vec2::new(0.5, 0.5), Vec2::new(0.0, -0.8)];
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
