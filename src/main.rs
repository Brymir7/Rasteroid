use conf::Conf;
use config::config::{ASTEROID_BASE_SPEED, PHYSICS_FRAME_TIME, WORLD_HEIGHT, WORLD_WIDTH};
use miniquad::*;
use glam::{vec3, vec4, Mat4, Quat, Vec2};
use std::{default, f32::consts::PI};
pub mod config;


trait Updatable {
    fn mut_pos(&mut self) -> &mut Vec2;
    fn mut_vel(&mut self) -> &mut Vec2;
    fn vel(&self) -> Vec2;
    fn update_pos(&mut self) {
        let vel = self.vel();
        let pos = self.mut_pos();
        pos.x += vel.x * PHYSICS_FRAME_TIME;
        pos.y += vel.y * PHYSICS_FRAME_TIME;
    }
    fn update(&mut self) {
        self.update_pos();
    }
}
struct BulletBlueprint {
    velocity: Vec2,
    pos: Vec2,
    damage: f32,
}
struct AsteroidBlueprint {
    velocity: Vec2,
    pos: Vec2,
    health: f32,
}
enum GameEventType {
    SpawnBullet(BulletBlueprint),
    SpawnAsteroid(Asteroid),
    PlayerHit,
    AsteroidKilled,
    PlayerPickupHealth(f32),
    PlayerPickupAmmo(i32),
    PlayerPickupWeapon(Weapon),
    PlayerShootBullet(BulletBlueprint),
    PlayerReload,
}

struct GameEvent {
    event_type: GameEventType,
    target: ObjectRef,
    triggered_by: ObjectRef,
}

struct World {
    player: Player,
    asteroids: Vec<Asteroid>,
    bullets: Vec<Bullet>,
    objects: Vec<Vec<ObjectRef>>, 
    default_pipeline: Pipeline,
    asteroid_pipeline: Pipeline,
}

impl World {
    fn new(ctx: &mut Context, default_pipeline: Pipeline, asteroid_pipeline:  Pipeline) -> Self {
        let player = Player::new(ctx);
        let asteroids = vec![];
        let bullets = vec![];
        let objects = vec![];
        World {
            player,
            asteroids,
            default_pipeline:  default_pipeline,
            asteroid_pipeline:  asteroid_pipeline,
            bullets,
            objects,
        }
    }

    fn update(&mut self, ctx: &mut Context) {
        if self.asteroids.len() < 1 {
            let asteroid = Asteroid::new(
                ctx, 
                Vec2::new(100.0, 150.0),
                Vec2::new(ASTEROID_BASE_SPEED, ASTEROID_BASE_SPEED),
                23.0,
            );
            self.asteroids.push(asteroid);
        }
        for asteroid in &mut self.asteroids {
            asteroid.update();
        }
        for bullet in &mut self.bullets {
            bullet.update();
        }
        let game_event = self.player.update();
        for event in game_event {
            match event.event_type {
                GameEventType::PlayerShootBullet(bullet) => {
                    self.bullets.push(Bullet::from_blueprint(ctx, bullet));
                }
                _ => {}
            }
        }
    }

    fn draw(&self, ctx: &mut Context) {
        ctx.apply_pipeline(&self.asteroid_pipeline);
        for asteroid in &self.asteroids {
            asteroid.draw(ctx);

        }
        ctx.apply_pipeline(&self.default_pipeline);
        for bullet in &self.bullets {
            bullet.draw(ctx);
        }
        self.player.draw(ctx);
    }
}
#[derive(Clone)]
enum WeaponType {
    Pistol,
    Shotgun,
    MachineGun,
}

#[derive(Clone)]
struct Weapon {
    weapon_type: WeaponType,
    ammo: i32,
    fire_rate: f32,
    bullet_speed: f32,
    damage: f32,
    reload_time: f32,
}

impl Weapon {
    fn default() -> Self {
        Weapon {
            weapon_type: WeaponType::Pistol,
            ammo: 10,
            fire_rate: 1.0,
            bullet_speed: 45.0,
            damage: 1.0,
            reload_time: 2.0,
        }
    }

    fn shoot_bullet(&mut self, origin: Vec2, direction: Vec2) -> BulletBlueprint {
        BulletBlueprint {
            velocity: direction.normalize() * self.bullet_speed,
            pos: origin,
            damage: self.damage,
        }
    }

}
#[derive(Clone)]
enum ObjectType {
    Weapon(Weapon),
    HealthPack(f32),
    AmmoPack(i32),
    Bullet,
    Player,
    Enemy,
    Asteroid,
}

enum ObjectRef {
    Player(usize),
    Asteroid(usize),
    Bullet(usize),
}
#[derive(Clone)]
struct Object {
    position: Vec2,
    object_type: ObjectType,
}
enum EventsByPlayer {
    ShootBullet(BulletBlueprint),
}

struct Player {
    object: Object,
    velocity: Vec2,
    hitbox: Rect,
    health: f32,
    vertices: Vec<Vec2>,
    vertex_buffer: BufferId,
    index_buffer: BufferId,
    looking_at: Vec2,
    weapon: Weapon,
    event_queue: Vec<EventsByPlayer>,
}

impl Updatable for Player {
    fn mut_pos(&mut self) -> &mut Vec2 {
        &mut self.object.position
    }

    fn mut_vel(&mut self) -> &mut Vec2 {
        &mut self.velocity
    }

    fn vel(&self) -> Vec2 {
        self.velocity
    }
}

impl Player {
    fn new(ctx: &mut Context) -> Self { 
        let vertices = vec![
            Vec2::new(-0.5, 0.5),
            Vec2::new(0.5, 0.5),
            Vec2::new(0.0, -0.8),
        ];
        let mut vertex_buffer_data: Vec<f32> = Vec::new();
        for vertex in &vertices {
            vertex_buffer_data.push(vertex.x);
            vertex_buffer_data.push(vertex.y);
            vertex_buffer_data.extend_from_slice(&[0.5, 0.5, 0.5, 1.0]);
        }

        Player {
            object: Object {
                position: Vec2::new(WORLD_WIDTH / 2.0, WORLD_HEIGHT / 2.0),
                object_type: ObjectType::Player,
            },
            velocity: Vec2::new(0.0, 0.0),
            hitbox: Rect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            vertex_buffer: ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&vertex_buffer_data),
            ),
            vertices,
            index_buffer: ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&[0, 1, 2]),
            ),
            health: 100.0,
            looking_at: Vec2::new(0.0, 0.0),
            weapon: Weapon::default(),
            event_queue: Vec::new(),
        }
    }

    fn handle_keyboard(&mut self, key_code: KeyCode ) {
        match key_code {
            KeyCode::W => self.velocity.y = -1.0,
            KeyCode::S => self.velocity.y = 1.0,
            KeyCode::A => self.velocity.x = -1.0,
            KeyCode::D => self.velocity.x = 1.0,
            KeyCode::Space => {
                if self.weapon.ammo < 0 {
                    return;
                }
                let bullet = self.weapon.shoot_bullet(self.object.position, self.looking_at);
                self.event_queue.push(EventsByPlayer::ShootBullet(bullet));
            }
            _ => {}
        }
    }
    fn handle_mouse(&mut self, x: f32, y: f32) {
        self.looking_at = Vec2::new(x - self.object.position.x, y - self.object.position.y);

    }
    fn update(&mut self) -> Vec<GameEvent> {
        Updatable::update(self);
        let mut events = Vec::new();
        for event in &self.event_queue {
            match event {
                EventsByPlayer::ShootBullet(bullet) => {
                    events.push(GameEvent {
                        event_type: GameEventType::PlayerShootBullet(
                            BulletBlueprint {
                                velocity: bullet.velocity,
                                pos: bullet.pos,
                                damage: bullet.damage,
                            }
                        ),
                        target: ObjectRef::Bullet(0),
                        triggered_by: ObjectRef::Player(0),
                    });
                }
            }
        }
        self.event_queue.clear();
        events
    }

    fn draw(&self, ctx: &mut Context) {
        let (width, height) = window::screen_size();
        let looking_at_normalized = self.looking_at.normalize();
        let rotation = Quat::from_rotation_arc(
            vec3(0.0, -1.0, 0.0),
            vec3(looking_at_normalized.x, looking_at_normalized.y, 0.0),
        );
        let model = Mat4::from_scale_rotation_translation(
            vec3(width / 8.0, height / 8.0, 1.0),
            rotation,
            vec3(self.object.position.x, self.object.position.y, 0.0)
        );
        let proj = Mat4::orthographic_rh_gl(0.0, width, height, 0.0 , -1.0, 1.0);
        let mvp = proj*model;
        let bindings = Bindings {
            vertex_buffers: vec![self.vertex_buffer.clone()],
            index_buffer: self.index_buffer.clone(),
            images: vec![],
        };
        ctx.apply_bindings(&bindings);
        ctx.apply_uniforms(UniformsSource::table(&mvp));
        ctx.draw(0, self.vertices.len() as i32, 1);
    }
}
struct CollisionResponse {
    collided: bool,
    new_velocity: Vec2,
    new_position: Vec2,
}
struct Rect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}
impl Rect {
    fn from_vertices(vertices: &Vec<Vec2>) -> Rect {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for vertex in vertices {
            if vertex.x < min_x {
                min_x = vertex.x;
            }
            if vertex.y < min_y {
                min_y = vertex.y;
            }
            if vertex.x > max_x {
                max_x = vertex.x;
            }
            if vertex.y > max_y {
                max_y = vertex.y;
            }
        }
        Rect {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }
    fn get_collision_response(&self, vel: Vec2, other: &Rect) -> CollisionResponse {
        let mut new_position = Vec2::new(self.x, self.y);
        let mut new_velocity = vel;
        let mut collided = false;

        let x_overlap = (self.x + self.width >= other.x) && (other.x + other.width >= self.x);
        let y_overlap = (self.y + self.height >= other.y) && (other.y + other.height >= self.y);

        if x_overlap && y_overlap {
            collided = true;
            let x_overlap = (self.x + self.width / 2.0) - (other.x + other.width / 2.0);
            let y_overlap = (self.y + self.height / 2.0) - (other.y + other.height / 2.0);

            if x_overlap.abs() > y_overlap.abs() {
                new_velocity.x = 0.0;
                if x_overlap > 0.0 {
                    new_position.x += x_overlap.abs();
                } else {
                    new_position.x -= x_overlap.abs();
                }
            } else {
                new_velocity.y = 0.0;
                if y_overlap > 0.0 {
                    new_position.y += y_overlap.abs();
                } else {
                    new_position.y -= y_overlap.abs();
                }
            }
        }
        CollisionResponse {
            collided,
            new_velocity,
            new_position,
        }
    }

}
struct Asteroid {
    object: Object,
    velocity: Vec2,
    health: f32,
    hitbox: Rect,
    vertices: Vec<Vec2>,
    vertex_buffer: BufferId,
    index_buffer: BufferId,
}

impl Updatable for Asteroid {
    fn mut_pos(&mut self) -> &mut Vec2 {
        &mut self.object.position
    }

    fn mut_vel(&mut self) -> &mut Vec2 {
        &mut self.velocity
    }

    fn vel(&self) -> Vec2 {
        self.velocity
    }
}
impl Asteroid {
    fn new(ctx: &mut Context, pos: Vec2, vel: Vec2, health: f32) -> Self {
        let vertices = Asteroid::gen_vertices(health);
        
        let mut buffer_data: Vec<f32> = Vec::new();
        for vertex in &vertices {
            buffer_data.push(vertex.x);
            buffer_data.push(vertex.y);
            buffer_data.extend_from_slice(&[0.5, 0.5, 0.5, 1.0]);
        }

        let vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&buffer_data),
        );

        let mut indices: Vec<u16> = Vec::new();
        for i in 1..(vertices.len() as u16 - 1) {
            indices.extend_from_slice(&[0, i, i + 1]);
        }
        let index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices),
        );

        Asteroid {
            object: Object {
                object_type: ObjectType::Asteroid,
                position: pos,
            },
            velocity: vel,
            health,
            hitbox: Rect::from_vertices(&vertices),
            vertices,
            vertex_buffer,
            index_buffer,
        }
    }

     fn gen_vertices(health: f32) -> Vec<Vec2> {
        let num_vertices = (health.ceil() as usize).max(3).min(12);
        let radius = 20.0 + health * 5.0; // Base size + scaling factor
        let mut vertices = Vec::with_capacity(num_vertices);

        for i in 0..num_vertices {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / (num_vertices as f32);
            let random_offset = rand::random::<f32>() * 0.4 + 0.8; // Random factor between 0.8 and 1.2
            let x = angle.cos() * radius * random_offset;
            let y = angle.sin() * radius * random_offset;
            vertices.push(Vec2::new(x, y));
        }

        vertices
    }

    pub fn update_health(&mut self, new_health: f32, ctx: &mut Context) {
        if new_health < self.health {
            self.health = new_health;
            self.remove_vertex(ctx);
        }
    }

    fn remove_vertex(&mut self, ctx: &mut Context) {
        if self.vertices.len() > 3 {
            let remove_index = rand::random::<usize>() % self.vertices.len();
            self.vertices.remove(remove_index);

            // Regenerate buffer data
            let mut buffer_data: Vec<f32> = Vec::new();
            for vertex in &self.vertices {
                buffer_data.push(vertex.x);
                buffer_data.push(vertex.y);
                buffer_data.extend_from_slice(&[0.5, 0.5, 0.5, 1.0]);
            }

            // Update vertex buffer
            ctx.buffer_update(self.vertex_buffer, BufferSource::slice(&buffer_data));

            // Regenerate indices
            let mut indices: Vec<u16> = Vec::new();
            for i in 1..(self.vertices.len() as u16 - 1) {
                indices.extend_from_slice(&[0, i, i + 1]);
            }

            // Update index buffer
            ctx.buffer_update(self.index_buffer, BufferSource::slice(&indices));
        }
    }

    fn draw(&self, ctx: &mut Context) {
        let (width, height) = window::screen_size();
        let model = Mat4::from_scale_rotation_translation(
            vec3(1.0, 1.0, 1.0),
            Quat::IDENTITY,
            vec3(self.object.position.x, self.object.position.y, 0.0)
        );
        let proj = Mat4::orthographic_rh_gl(0.0, width, height, 0.0 , -1.0, 1.0);
        let mvp = proj*model;
        let bindings = Bindings {
            vertex_buffers: vec![self.vertex_buffer.clone()],
            index_buffer: self.index_buffer.clone(),
            images: vec![],
        };
        ctx.apply_bindings(&bindings);
        ctx.apply_uniforms(UniformsSource::table(&mvp));
        ctx.draw(0, self.vertices.len() as i32, 1);
    }
    fn update(&mut self) {
        Updatable::update(self);
    }
}
#[derive(Clone)]
struct Bullet {
    object: Object,
    velocity: Vec2,
    vertices: Vec<Vec2>,
    vertex_buffer: BufferId,
    index_buffer: BufferId,
}

impl Updatable for Bullet {
    fn mut_pos(&mut self) -> &mut Vec2 {
        &mut self.object.position
    }

    fn mut_vel(&mut self) -> &mut Vec2 {
        &mut self.velocity
    }

    fn vel(&self) -> Vec2 {
        self.velocity
    }
}

impl Bullet {
    fn new(ctx: &mut Context, pos: Vec2, velocity: Vec2) -> Bullet {
        let vertices = vec![
            Vec2::new(-0.5, 0.5),
            Vec2::new(0.5, 0.5),
            Vec2::new(0.5, -0.5),
            Vec2::new(-0.5, -0.5),
        ];
        let mut vertex_buffer_data: Vec<f32> = Vec::new();
        for vertex in &vertices {
            vertex_buffer_data.push(vertex.x);
            vertex_buffer_data.push(vertex.y);
            vertex_buffer_data.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        }

        Bullet {
            object: Object {
                position: pos,
                object_type: ObjectType::Bullet,
            },
            velocity,
            vertices,
            vertex_buffer: ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&vertex_buffer_data),
            ),
            index_buffer: ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&[0, 1, 2, 0, 2, 3]),
            ),

        }
    }
    fn from_blueprint(ctx: &mut Context, blueprint: BulletBlueprint) -> Bullet {

        Bullet::new(ctx, blueprint.pos, blueprint.velocity)
    }
    fn update(&mut self) {

        Updatable::update(self);
    }
    fn draw(&self, ctx: &mut Context) {
        let (width, height) = window::screen_size();
        let model = Mat4::from_scale_rotation_translation(
            vec3(25.0, 25.0, 1.0),
            Quat::IDENTITY,
            vec3(self.object.position.x, self.object.position.y, 0.0)
        );
        let proj = Mat4::orthographic_rh_gl(0.0, width, height, 0.0 , -1.0, 1.0);
        let mvp = proj*model;
        let bindings = Bindings {
            vertex_buffers: vec![self.vertex_buffer.clone()],
            index_buffer: self.index_buffer.clone(),
            images: vec![],
        };
        ctx.apply_bindings(&bindings);
        ctx.apply_uniforms(UniformsSource::table(&mvp));
        ctx.draw(0, self.vertices.len() as i32, 1);
    }
}

struct Stage {
    world: World,
    elapsed_time: f32,
    ctx: Box<dyn RenderingBackend>,

    last_time: f64,
}

impl Stage {
    fn load_shaders(ctx: &mut Context) -> Vec<Pipeline> {
        let asteroid_shader = ctx.new_shader(
            ShaderSource::Glsl {
                vertex: include_str!("asteroid_vertex.glsl"),
                fragment: include_str!("asteroid_fragment.glsl"),
            },
            ShaderMeta { uniforms: 
                UniformBlockLayout {
                    uniforms: vec![
                        UniformDesc::new("mvp", UniformType::Mat4),
                    ],
                }, images: vec![] },
        ).expect("Failed to load asteroid shader.");
        let default_shader = ctx.new_shader(
            ShaderSource::Glsl {
                vertex: include_str!("vertex.glsl"),
                fragment: include_str!("fragment.glsl"),
            },
            ShaderMeta {
                uniforms: UniformBlockLayout {
                    uniforms: vec![
                        UniformDesc::new("mvp", UniformType::Mat4),
                    ],
                },
                images: vec![],
            },
        ).expect("Failed to load default shader.");
        return vec![        ctx.new_pipeline(
            &[
                BufferLayout {
                    stride: 24,
                    ..Default::default()
                }
            ],
            &[
                VertexAttribute::new("pos", VertexFormat::Float2),
                VertexAttribute::new("color0", VertexFormat::Float4),
            ],
            default_shader,
            PipelineParams::default(),
        ), ctx.new_pipeline(
            
            &[BufferLayout {
                stride: 24,
                ..Default::default()
            }],
            &[
                VertexAttribute::new("pos", VertexFormat::Float2),
                VertexAttribute::new("color0", VertexFormat::Float4),
            ],
            asteroid_shader,
            PipelineParams::default(),
        )];
    }

    fn new() -> Self {
        let mut ctx = window::new_rendering_backend();
        let gfx_pipelines = Stage::load_shaders(&mut *ctx);
        let  world = World::new(&mut *ctx, gfx_pipelines[0], gfx_pipelines[1]);
        Stage {
            world,
            elapsed_time: 0.0,
            ctx: ctx,
            last_time: date::now(),
        }
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {
        let delta = date::now() - self.last_time;
        self.elapsed_time += (date::now() - self.last_time) as f32;
        self.last_time = date::now();
        while self.elapsed_time >= PHYSICS_FRAME_TIME {
            self.world.update(&mut *self.ctx);
            self.elapsed_time -= PHYSICS_FRAME_TIME;

        }

    }

    fn draw(&mut self) {
        self.ctx.clear(Some((1.0, 1.0, 1.0, 1.0)), None, None);
        match self.ctx.info().backend {
            Backend::OpenGl  => {
                self.world.draw(&mut *self.ctx);
            }
            _ => {}
        }
    }
    fn key_down_event(&mut self, keycode: KeyCode, _keymods: KeyMods, _repeat: bool) {
        self.world.player.handle_keyboard(keycode);
    }
    fn mouse_motion_event(&mut self, x: f32, y: f32) {
        self.world.player.handle_mouse(x, y);
    }
}

fn main() {
    miniquad::start(Conf {
        window_title: "Rasteroid".to_owned(),
        window_width: WORLD_WIDTH as i32,
        window_height: WORLD_HEIGHT as i32,
        fullscreen: false,
        ..Default::default()
    }, || Box::new(Stage::new()));
}
