use macroquad::{prelude::*};


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
    fn update(&mut self ) {
        self.player.update();
        for asteroid in &mut self.asteroids {
            asteroid.update();
        }
        for bullet in &mut self.bullets {
            bullet.update();
        }
        for object in objects {
            object.update();
        }
    }
    fn draw(&self) {
        self.player.draw();
        for asteroid in &self.asteroids {
            asteroid.draw();
        }
        for bullet in &self.bullets {
            bullets.draw();
        }
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
enum Direction {

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
    fn shoot_bullet(&mut self, origin: Vec2, direction: Vec2) -> Bullet  {
        Bullet {
            object: Object {
                position: origin,
                object_type: ObjectType::Bullet,
            },
            velocity: Vec2::new(direction.x.signum() * self.bullet_speed, direction.y.signum() * self.bullet_speed)
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
    Asteroid
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
struct Asteroid{
    object: Object,
    velocity: Vec2,
    health: f32,
}
impl Asteroid {
    fn new(pos: Vec2, vel: Vec2, health: f32) -> Self {
        Asteroid{
            object: Object {
                object_type: ObjectType::Asteroid,
                position: pos,
            },
            velocity: vel,
            health: health
        }
    }
}
struct Bullet {
    object: Object,
    
    velocity: Vec2,
}
impl Bullet {
    fn new(pos: Vec2, velocity: Vec2) -> Bullet {
        Bullet {
            object: Object {
                position: pos,
                object_type: ObjectType::Bullet,
            },
            velocity: velocity,
        }
    }
    fn update(&mut self) {}
}


#[macroquad::main("Hello world")]
async fn main() {
    loop {
        clear_background(WHITE);
        draw_text("Hello, world!", 20.0, 20.0, 30.0, BLACK);
        next_frame().await
    }
}