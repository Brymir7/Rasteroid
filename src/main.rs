use conf::Conf;
use miniquad::*;
use glam::{vec3, vec4, Mat4, Quat, Vec2};
use std::f32::consts::PI;

pub mod config {
    pub const PHYSICS_FRAME_TIME: f32 = 1.0 / 60.0;
    pub const WORLD_WIDTH: f32 = 800.0;
    pub const WORLD_HEIGHT: f32 = 600.0;
}

trait Updatable {
    fn mut_pos(&mut self) -> &mut Vec2;
    fn mut_vel(&mut self) -> &mut Vec2;
    fn vel(&self) -> Vec2;
    fn update_pos(&mut self) {
        let vel = self.vel();
        let pos = self.mut_pos();
        pos.x += vel.x * config::PHYSICS_FRAME_TIME;
        pos.y += vel.y * config::PHYSICS_FRAME_TIME;
    }
    fn update(&mut self) {
        self.update_pos();
    }
}

enum GameEventType {
    AsteroidKilled,
    PlayerHit,
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
}

impl World {
    fn new() -> Self {
        let player = Player {
            object: Object {
                position: Vec2::new(0.0, 0.0),
                object_type: ObjectType::Player,
            },
            velocity: Vec2::new(0.0, 0.0),
            health: 100.0,
            looking_at: Vec2::new(0.0, 0.0),
            weapon: Weapon::default(),
        };
        let asteroids = vec![];
        let bullets = vec![];
        let objects = vec![];
        World {
            player,
            asteroids,
            bullets,
            objects,
        }
    }

    fn update(&mut self, ctx: &mut Context) {
        if self.asteroids.len() < 1 {
            let asteroid = Asteroid::new(
                ctx, 
                Vec2::new(100.0, 150.0),
                Vec2::new(1.0, 1.0),
                1.0,
            );
            self.asteroids.push(asteroid);
        }
        for asteroid in &mut self.asteroids {
            asteroid.update();
        }
        for bullet in &mut self.bullets {
            bullet.update();
        }
        self.player.update();
    }

    fn draw(&self, ctx: &mut Context, pipeline: &Pipeline) {
        ctx.apply_pipeline(&pipeline);
        for asteroid in &self.asteroids {
            asteroid.draw(ctx, pipeline);

        }
        self.player.draw(ctx);
    }
}

enum WeaponType {
    Pistol,
    Shotgun,
    MachineGun,
}

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
            bullet_speed: 1.0,
            damage: 1.0,
            reload_time: 2.0,
        }
    }

    fn shoot_bullet(&mut self, origin: Vec2, direction: Vec2) -> Bullet {
        Bullet {
            object: Object {
                position: origin,
                object_type: ObjectType::Bullet,
            },
            velocity: direction.normalize() * self.bullet_speed,
        }
    }

}

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

struct Object {
    position: Vec2,
    object_type: ObjectType,
}

struct Player {
    object: Object,
    velocity: Vec2,
    health: f32,
    looking_at: Vec2,
    weapon: Weapon,
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
    fn handle_keyboard(&mut self, key_code: KeyCode ) {
        match key_code {
            KeyCode::W => self.velocity.y = -1.0,
            KeyCode::S => self.velocity.y = 1.0,
            KeyCode::A => self.velocity.x = -1.0,
            KeyCode::D => self.velocity.x = 1.0,
            _ => {}
        }

        
    }
    fn handle_mouse(&mut self, x: f32, y: f32) {
        self.looking_at = Vec2::new(x - self.object.position.x, y - self.object.position.y);

    }
    fn update(&mut self) {
        Updatable::update(self);
    }

    fn draw(&self, ctx: &mut Context) {
        // Draw the player as a simple triangle or circle
    }
}

struct Asteroid {
    object: Object,
    velocity: Vec2,
    health: f32,
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
            // Add color data (e.g., grey)
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
            vertices,
            vertex_buffer,
            index_buffer,
        }
    }

    fn gen_vertices(health: f32) -> Vec<Vec2> {
        let num_edges = (3.0 + health * 5.0) as usize;
        let mut edges = Vec::with_capacity(num_edges);
        for i in 0..num_edges {
            let angle = i as f32 / num_edges as f32 * PI * 2.0;
            let radius = 3.0 + health;
            edges.push(Vec2::new(angle.cos() * radius, angle.sin() * radius));
        }
        println!("Generated asteroid with {:?} edges", edges);

        edges
        // return vec![
        //     Vec2::new(-0.5, 0.5),
        //     Vec2::new(0.5, 0.5),
        //     Vec2::new(0.0, -0.5),
        // ]
    }
    fn draw(&self, ctx: &mut Context, pipeline: &Pipeline) {
        let model = Mat4::from_scale_rotation_translation(
            vec3(15.0, 15.0, 1.0),
            Quat::IDENTITY,
            vec3(self.object.position.x, self.object.position.y, 0.0)
        );
        let (width, height) = window::screen_size();
        let proj = Mat4::orthographic_rh_gl(0.0, width, height, 0.0 , -1.0, 1.0);
        let mvp = proj*model;
        println!("Drawing vertices at {:?}", mvp* vec4(self.vertices[0].x, self.vertices[0].y, 0.0, 1.0));
        let bindings = Bindings {
            vertex_buffers: vec![self.vertex_buffer.clone()],
            index_buffer: self.index_buffer.clone(),
            images: vec![],
        };

        ctx.apply_pipeline(pipeline); // Assuming you have a default pipeline set up
        ctx.apply_bindings(&bindings);
        ctx.apply_uniforms(UniformsSource::table(&mvp));
        ctx.draw(0, self.vertices.len() as i32 - 2, 1);
    }
    fn update(&mut self) {
        Updatable::update(self);
    }
}

struct Bullet {
    object: Object,
    velocity: Vec2,
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
    fn new(pos: Vec2, velocity: Vec2) -> Bullet {
        Bullet {
            object: Object {
                position: pos,
                object_type: ObjectType::Bullet,
            },
            velocity,
        }
    }

    fn update(&mut self) {
        Updatable::update(self);
    }
}

struct Stage {
    world: World,
    elapsed_time: f32,
    ctx: Box<dyn RenderingBackend>,
    asteroid_pipeline: Pipeline,
    last_time: f64,

}

impl Stage {
    fn new() -> Self {
        let  world = World::new();
        let mut ctx = window::new_rendering_backend();

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

        let asteroid_pipeline = ctx.new_pipeline(
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
        );

        Stage {
            world,
            elapsed_time: 0.0,
            ctx: ctx,
            asteroid_pipeline,
            last_time: date::now(),
        }
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {
        self.elapsed_time += (date::now() - self.last_time) as f32;
        self.last_time = date::now();
        while self.elapsed_time >= config::PHYSICS_FRAME_TIME {
            // self.world.player.handle_input();
            self.world.update(&mut *self.ctx);
            self.elapsed_time -= config::PHYSICS_FRAME_TIME;
        }
    }

    fn draw(&mut self) {
        self.ctx.clear(Some((1.0, 1.0, 1.0, 1.0)), None, None);
        match self.ctx.info().backend {
            Backend::OpenGl  => {
                self.world.draw(&mut *self.ctx, &self.asteroid_pipeline);
            }
            _ => {}
        }
    }
    // fn key_down_event(&mut self, _keycode: KeyCode, _keymods: KeyMods, _repeat: bool) {
    //     self.world.player.handle_input();
    // }
    // fn mouse_motion_event(&mut self, _x: f32, _y: f32) {
    //     self.world.player.handle_input();
    // }
}

fn main() {
    miniquad::start(Conf {
        window_title: "Rasteroid".to_owned(),
        window_width: config::WORLD_WIDTH as i32,
        window_height: config::WORLD_HEIGHT as i32,
        fullscreen: false,
        ..Default::default()
    }, || Box::new(Stage::new()));
}
