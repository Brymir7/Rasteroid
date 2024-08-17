use conf::Conf;
use config::config::{ ASTEROID_BASE_SPEED, DEFAULT_SCALE_FACTTOR_X, DEFAULT_SCALE_FACTTOR_Y, PHYSICS_FRAME_TIME, PLAYER_HITBOX_SIZE, TILE_SIZE, WORLD_HEIGHT, WORLD_WIDTH };
use miniquad::*;
use glam::{ vec3, Mat4, Quat, Vec2 };
pub mod config;

trait Updatable {
    fn mut_pos(&mut self) -> &mut Vec2;
    fn mut_vel(&mut self) -> &mut Vec2;
    fn vel(&self) -> Vec2;
    fn get_hitbox(&self) -> Rect;
    fn update_pos(&mut self) {
        let vel = self.vel();
        let pos = self.mut_pos();
        pos.x += vel.x * PHYSICS_FRAME_TIME;
        pos.y += vel.y * PHYSICS_FRAME_TIME;
    }
    fn bounds_to_keep_pos_in(&self) -> Rect {
        Rect {
            x: 0.0,
            y: 0.0,
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
        }
    }
    fn stay_in_bounds(&mut self) {
        let world_bounds = self.bounds_to_keep_pos_in();
        let hitbox = self.get_hitbox();

        if hitbox.x < world_bounds.x {
            self.mut_pos().x = world_bounds.x + hitbox.width / 2.0;
            self.mut_vel().x = 0.0;
        }
        if hitbox.y < world_bounds.y {
            self.mut_pos().y = world_bounds.y + hitbox.height / 2.0;
            self.mut_vel().y = 0.0;
        }
        if hitbox.x + hitbox.width > world_bounds.x + world_bounds.width {
            self.mut_pos().x = world_bounds.x + world_bounds.width - hitbox.width / 2.0;
            self.mut_vel().x = 0.0;

        }
        if hitbox.y + hitbox.height > world_bounds.y + world_bounds.height {
            self.mut_pos().y = world_bounds.y + world_bounds.height - hitbox.height / 2.0;
            self.mut_vel().y = 0.0;
        }
    }
    fn update(&mut self) -> Option<GameEvent> {
        self.update_pos();
        self.stay_in_bounds();
        None
    }
}
trait Collision {
    fn get_collision_handler (&self, obj_type: &ObjectType) -> Option<Box<dyn CollisionHandler>>;
    fn vel(&self) -> Vec2;
    fn get_hitbox(&self) -> Rect;
    fn check_collision(&self, other_hitbox: &Rect) -> CollisionResponse {
        let self_hitbox = self.get_hitbox();
        self_hitbox.get_collision_response(self.vel(), &other_hitbox)
    }
    fn check_and_handle_collision(&mut self, other: &Object, other_hitbox: &Rect) -> Option<GameEvent> {
        let response = self.check_collision(other_hitbox);
        if let Some(collision_handler) = self.get_collision_handler(&other.object_type) {
            return collision_handler.handle_collision_response(response, Some(other));
        } else {
            None
        }
    }

}
trait CollisionHandler {
    fn handle_collision_response(
        &self,
        response: CollisionResponse,
        other: Option<&Object>
    ) -> Option<GameEvent>;
}
struct PlayerAsteroidCollisionHandler;
impl CollisionHandler for PlayerAsteroidCollisionHandler {
    fn handle_collision_response(
        & self,
        response: CollisionResponse,
        other: Option<&Object>
    ) -> Option<GameEvent> {
        if response.collided {
            match other {
                Some(other) => {
                    match other.object_type {
                        ObjectType::Asteroid => {
                            return Some(GameEvent {
                                event_type: GameEventType::PlayerHit,
                                target: ObjectRef::Player,
                                triggered_by: ObjectRef::Asteroid(0),
                            });
                        }
                        _ => {}
                    }
                }
                None => {}
            }
        }
        None
    }
}

struct AsteroidBulletCollisionHandler;
impl CollisionHandler for AsteroidBulletCollisionHandler {
    fn handle_collision_response(
        &self,
        response: CollisionResponse,
        other: Option<&Object>
    ) -> Option<GameEvent> {
        if response.collided {
            match other {
                Some(other) => {
                    match other.object_type {

                        ObjectType::Bullet => {
                            println!("Asteroid hit!");
                            return Some(GameEvent {
                                event_type: GameEventType::AsteroidKilled,
                                target: ObjectRef::Asteroid(0),
                                triggered_by: ObjectRef::Bullet(0),
                            });
                        }
                        _ => {}
                    }
                }
                None => {}
            }
        }
        None
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
    tiles_of_objects: Vec<Vec<ObjectRef>>,
    default_pipeline: Pipeline,
    asteroid_pipeline: Pipeline,
}

impl World {
    fn new(ctx: &mut Context, default_pipeline: Pipeline, asteroid_pipeline: Pipeline) -> Self {
        let player = Player::new(ctx);
        let asteroids = vec![];
        let bullets = vec![];
        let num_tiles_x = (WORLD_WIDTH / TILE_SIZE) as usize;
        let num_tiles_y = (WORLD_HEIGHT / TILE_SIZE) as usize;
        let tiles_of_objects: Vec<Vec<ObjectRef>> = (0..num_tiles_y)
    .map(|_| vec![ObjectRef::None; num_tiles_x])
    .collect();
        World {
            player,
            asteroids,
            default_pipeline: default_pipeline,
            asteroid_pipeline: asteroid_pipeline,
            bullets,
            tiles_of_objects,
        }
    }
    fn get_objects_index(&self, object: &Object) -> (usize, usize) {
        let x = (object.position.x / TILE_SIZE).round() as usize;
        let y = (object.position.y / TILE_SIZE).round() as usize;
        (x, y)
    }

    fn add_object_to_tile(&mut self, ctx: &mut Context, object: &Object) {
        match object.object_type {
            ObjectType::Asteroid => {
                let (x, y) = self.get_objects_index(&object);
                self.tiles_of_objects[y][x] = ObjectRef::Asteroid(self.asteroids.len());
            }
            ObjectType::Bullet => {
                let (x, y) = self.get_objects_index(&object);
                self.tiles_of_objects[y][x] = ObjectRef::Bullet(self.bullets.len());
            }
            ObjectType::Player => {
                let (x, y) = self.get_objects_index(&object);
                self.tiles_of_objects[y][x] = ObjectRef::Player;
            }
            _ => {}
        }
    }
    fn remove_object_from_tile(&mut self, object: &Object) {
        let (x, y) = self.get_objects_index(object);
        self.tiles_of_objects[y][x] = ObjectRef::None;
    }
    fn handle_game_event(&mut self, ctx: &mut Context, event: GameEvent) {
        match event.event_type { 
            GameEventType::PlayerShootBullet(bullet)  => {
                self.add_object_to_tile(ctx, &Object {
                    position: bullet.pos,
                    object_type: ObjectType::Bullet,
                });
                self.bullets.push(Bullet::from_blueprint(ctx, bullet));
            }
            GameEventType::SpawnAsteroid(asteroid) => {
                self.add_object_to_tile(ctx, &asteroid.object);
                self.asteroids.push(asteroid);    
            }
            GameEventType::PlayerHit => {
                self.player.health -= 10.0;
                println!("Player hit! Health: {}", self.player.health);
            }
            GameEventType::AsteroidKilled => {
                match event.target {
                    ObjectRef::Asteroid(index) => {
                        let asteroid = self.asteroids.remove(index);
                        self.remove_object_from_tile(&asteroid.object);
                } _ => {} }
            }
            GameEventType::PlayerPickupHealth(health) => {
                self.player.health += health;
            }
            GameEventType::PlayerPickupAmmo(ammo) => {
                self.player.weapon.ammo += ammo;
            }
            GameEventType::PlayerPickupWeapon(weapon) => {
                self.player.weapon = weapon;
            }
            GameEventType::PlayerReload => {
                self.player.weapon.ammo = 10;
            }
            _ => {}
        }
    }
    fn update_tiles(&mut self, old_positions: Vec<Vec2>, new_positions: Vec<Vec2>) {
        assert!(old_positions.len() == new_positions.len());
        for i in 0..old_positions.len() {
            let old_pos_index = old_positions[i];
            let new_pos_index = new_positions[i];
            if old_pos_index != new_pos_index {
                let old_ref = self.tiles_of_objects[old_pos_index.y as usize][old_pos_index.x as usize];
                self.tiles_of_objects[old_pos_index.y as usize][old_pos_index.x as usize] = ObjectRef::None;
                self.tiles_of_objects[new_pos_index.y as usize][new_pos_index.x as usize] = old_ref;
            }
        }
    }
    fn get_surrounding_objects (&self, object: &Object) -> Vec<SurroundingObject> {
        let (x, y) = self.get_objects_index(object);
        let mut objects = Vec::new();
        for i in -1..2 {
            for j in -1..2 {
                let x = x as i32 + i;
                let y = y as i32 + j;
                if x >= 0 && x < self.tiles_of_objects[0].len() as i32 && y >= 0 && y < self.tiles_of_objects.len() as i32 {
                    objects.push(self.tiles_of_objects[y as usize][x as usize]);
                }
            }
        }
        objects.into_iter().filter_map(|object_ref| {
            match object_ref {
                ObjectRef::Asteroid(index) => {
                    Some(SurroundingObject {
                        object: self.asteroids[index].object.clone(),
                        hitbox: self.asteroids[index].get_hitbox(),
                    })
                }
                ObjectRef::Bullet(index) => {
                    Some(SurroundingObject {
                        object: self.bullets[index].object.clone(),
                        hitbox: self.bullets[index].get_hitbox(),
                    })
                }
                ObjectRef::Player => {
                    Some(SurroundingObject {
                        object: self.player.object.clone(),
                        hitbox: self.player.get_hitbox(),
                    })
                }
                _ => None,
            }
        }).collect()
    }
    fn update(&mut self, ctx: &mut Context) {
        let mut old_positions = Vec::new();
        let mut new_positions = Vec::new();
        if self.asteroids.len() < 1 {
            let asteroid = Asteroid::new(
                ctx,
                Vec2::new(100.0, 150.0),
                Vec2::new(ASTEROID_BASE_SPEED, ASTEROID_BASE_SPEED),
                23.0
            );
            self.add_object_to_tile(ctx, 
                &Object {
                    position: asteroid.object.position,
                    object_type: ObjectType::Asteroid,
                }
            );
            self.asteroids.push(asteroid);
        }
        let asteroids_len = self.asteroids.len();
        let mut asteroid_events = Vec::new();
        for i in 0..asteroids_len {  

            let surrounding_objects = self.get_surrounding_objects(&self.asteroids[i].object);
            let asteroid = &mut self.asteroids[i];
            
            let old_pos = asteroid.object.position;

            let game_event = asteroid.update(surrounding_objects);
            if let Some(event) = game_event {
                asteroid_events.push(event);
            }
            let new_pos = asteroid.object.position;
            let old_pos_index = old_pos / TILE_SIZE;
            let new_pos_index = new_pos / TILE_SIZE;
            if old_pos_index != new_pos_index {
                old_positions.push(old_pos_index);
                new_positions.push(new_pos_index);
            }
        }
        for event in asteroid_events {
            self.handle_game_event(ctx, event);
        }

        for bullet in &mut self.bullets {
            let old_pos = bullet.object.position;
            bullet.update();
            let new_pos = bullet.object.position;
            let old_pos_index = old_pos / TILE_SIZE;
            let new_pos_index = new_pos / TILE_SIZE;
            if old_pos_index != new_pos_index {
                old_positions.push(old_pos_index);
                new_positions.push(new_pos_index);
            }
        }
        let old_pos = self.player.object.position;
        let surrounding_objects = self.get_surrounding_objects(&self.player.object);
        let game_event = self.player.update(surrounding_objects);
        let new_pos = self.player.object.position;
        let old_pos_index = old_pos / TILE_SIZE;
        let new_pos_index = new_pos / TILE_SIZE;
        if old_pos_index != new_pos_index {
            old_positions.push(old_pos_index);
            new_positions.push(new_pos_index);
        }
        for event in game_event {
            self.handle_game_event(ctx, event);
        }
        self.update_tiles(old_positions, new_positions);
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
#[derive(Clone, Copy)]
enum ObjectRef {
    Player,
    Asteroid(usize),
    Bullet(usize),
    None,
}
#[derive(Clone)]
struct Object {
    position: Vec2,
    object_type: ObjectType,
}
enum EventsByPlayer {
    ShootBullet(BulletBlueprint),
}
struct SurroundingObject {
    object: Object,
    hitbox: Rect,
}
struct Player {
    object: Object,
    velocity: Vec2,
    health: f32,
    vertices: Vec<Vec2>,
    vertex_buffer: BufferId,
    index_buffer: BufferId,
    looking_at: Vec2,
    weapon: Weapon,
    player_event_queue: Vec<EventsByPlayer>,
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
    fn get_hitbox(&self) -> Rect {
        self.get_hitbox()
    }
}
impl Collision for Player {
    fn vel(&self) -> Vec2 {
        self.velocity
    }
    fn get_hitbox(&self) -> Rect {
        self.get_hitbox()
    }
    fn get_collision_handler(&self, obj_type: &ObjectType) -> Option<Box<dyn CollisionHandler>> {
        match obj_type {
            ObjectType::Asteroid => Some(Box::new(PlayerAsteroidCollisionHandler)),
            _ => None,
        }
    }
}
impl Player {
    fn new(ctx: &mut Context) -> Self {
        let vertices = vec![Vec2::new(-0.5, 0.5), Vec2::new(0.5, 0.5), Vec2::new(0.0, -0.8)];
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

            vertex_buffer: ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&vertex_buffer_data)
            ),
            vertices,
            index_buffer: ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&[0, 1, 2])
            ),
            health: 100.0,
            looking_at: Vec2::new(WORLD_WIDTH / 2.0, WORLD_HEIGHT / 2.0),
            weapon: Weapon::default(),
            player_event_queue: Vec::new(),
        }
    }
    fn get_hitbox (&self) -> Rect {
        Rect {
            x: self.object.position.x - PLAYER_HITBOX_SIZE / 2.0,
            y: self.object.position.y - PLAYER_HITBOX_SIZE / 2.0,
            width: PLAYER_HITBOX_SIZE,
            height: PLAYER_HITBOX_SIZE,
        }
    }

    fn handle_keyboard(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::W => {
                self.velocity.y += -1.0;
            }
            KeyCode::S => {
                self.velocity.y += 1.0;
            }
            KeyCode::A => {
                self.velocity.x += -1.0;
            }
            KeyCode::D => {
                self.velocity.x += 1.0;
            }
            KeyCode::Space => {
                if self.weapon.ammo < 0 {
                    return;
                }
                let looking_at_normalized = self.looking_at.normalize();
                let bullet = self.weapon.shoot_bullet(self.object.position+looking_at_normalized*PLAYER_HITBOX_SIZE*3.3, self.looking_at);
                self.player_event_queue.retain(|event| {
                    match event {
                        EventsByPlayer::ShootBullet(_) => true,
                    }
                });
                self.player_event_queue.push(EventsByPlayer::ShootBullet(bullet));
            }
            _ => {}
        }
    }
    fn handle_mouse(&mut self, x: f32, y: f32) {
        self.looking_at = Vec2::new(x - self.object.position.x, y - self.object.position.y);
    }
    fn update(&mut self, surrounding_objects: Vec<SurroundingObject>) -> Vec<GameEvent> {
        Updatable::update(self);
        let mut events = Vec::new();
        for object in surrounding_objects {
            let event = self.check_and_handle_collision(&object.object, &object.hitbox);
            if let Some(event) = event {
                events.push(event);
            }
        }
        for event in &self.player_event_queue {
            match event {
                EventsByPlayer::ShootBullet(bullet) => {
                    events.push(GameEvent {
                        event_type: GameEventType::PlayerShootBullet(BulletBlueprint {
                            velocity: bullet.velocity,
                            pos: bullet.pos,
                            damage: bullet.damage,
                        }),
                        target: ObjectRef::Bullet(0),
                        triggered_by: ObjectRef::Player,
                    });
                }
                
            }
        }
        self.player_event_queue.clear();
        events
    }

    fn draw(&self, ctx: &mut Context) {
        let (width, height) = window::screen_size();
        let looking_at_normalized = self.looking_at.normalize();
        let rotation = Quat::from_rotation_arc(
            vec3(0.0, -1.0, 0.0),
            vec3(looking_at_normalized.x, looking_at_normalized.y, 0.0)
        );
        let model = Mat4::from_scale_rotation_translation(
            vec3(DEFAULT_SCALE_FACTTOR_X, DEFAULT_SCALE_FACTTOR_Y, 1.0),
            rotation,
            vec3(self.object.position.x, self.object.position.y, 0.0)
        );
        let proj = Mat4::orthographic_rh_gl(0.0, width, height, 0.0, -1.0, 1.0);
        let mvp = proj * model;
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
#[derive(Clone, Copy, Debug)]
struct Rect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}
impl Rect {
    fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }
    fn get_collision_response(&self, vel: Vec2, other: &Rect) -> CollisionResponse {
        let mut new_position = Vec2::new(self.x, self.y);
        let mut new_velocity = vel;
        let mut collided = false;
        let self_center = self.center();
        let other_center = other.center();
        let x_overlap = self_center.x - other_center.x + (self.width + other.width) / 2.0;
        let y_overlap = self_center.y - other_center.y + (self.height + other.height) / 2.0;
        if x_overlap > 0.0 && y_overlap > 0.0 {
            if x_overlap < y_overlap {
                if vel.x > 0.0 {
                    new_position.x = other.x - self.width;
                } else {
                    new_position.x = other.x + other.width;
                }
                new_velocity.x = -vel.x;
            } else {
                if vel.y > 0.0 {
                    new_position.y = other.y - self.height;
                } else {
                    new_position.y = other.y + other.height;
                }
                new_velocity.y = -vel.y;
            }
            collided = true;
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
    base_radius: f32,
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

    fn get_hitbox(&self) -> Rect {
        self.get_hitbox()
    }
}
impl Collision for Asteroid {
    fn vel(&self) -> Vec2 {
        self.velocity
    }
    fn get_hitbox(&self) -> Rect {
        self.get_hitbox()
    }
    fn get_collision_handler(&self, obj_type: &ObjectType) -> Option<Box<dyn CollisionHandler>> {
        match obj_type {
            ObjectType::Bullet => Some(Box::new(AsteroidBulletCollisionHandler)),
            _ => None,
        }
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
            BufferSource::slice(&buffer_data)
        );

        let mut indices: Vec<u16> = Vec::new();
        for i in 1..(vertices.len() as u16) - 1 {
            indices.extend_from_slice(&[0, i, i + 1]);
        }
        let index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices)
        );

        Asteroid {
            object: Object {
                object_type: ObjectType::Asteroid,
                position: pos,
            },
            velocity: vel,
            health,
            base_radius: 20.0,
            vertices,
            vertex_buffer,
            index_buffer,
        }
    }
    fn get_current_radius(&self) -> f32 {
        self.base_radius * (self.health / 100.0).max(0.2)
    }

    fn gen_vertices(health: f32) -> Vec<Vec2> {
        let num_vertices = (health.ceil() as usize).max(3).min(12);
        let radius = 20.0 + health * 5.0; // Base size + scaling factor
        let mut vertices = Vec::with_capacity(num_vertices);

        for i in 0..num_vertices {
            let angle = (2.0 * std::f32::consts::PI * (i as f32)) / (num_vertices as f32);
            let random_offset = rand::random::<f32>() * 0.4 + 0.8; // Random factor between 0.8 and 1.2
            let x = angle.cos() * radius * random_offset;
            let y = angle.sin() * radius * random_offset;
            vertices.push(Vec2::new(x, y));
        }

        return vec![Vec2::new(-0.5, 0.5), Vec2::new(0.5, 0.5), Vec2::new(0.0, -0.8)];
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

            let mut buffer_data: Vec<f32> = Vec::new();
            for vertex in &self.vertices {
                buffer_data.push(vertex.x);
                buffer_data.push(vertex.y);
                buffer_data.extend_from_slice(&[0.5, 0.5, 0.5, 1.0]);
            }

            ctx.buffer_update(self.vertex_buffer, BufferSource::slice(&buffer_data));

            let mut indices: Vec<u16> = Vec::new();
            for i in 1..(self.vertices.len() as u16) - 1 {
                indices.extend_from_slice(&[0, i, i + 1]);
            }

            ctx.buffer_update(self.index_buffer, BufferSource::slice(&indices));
        }
    }
    fn get_hitbox(&self) -> Rect {
        let current_radius = self.get_current_radius();
        Rect {
            x: self.object.position.x - current_radius,
            y: self.object.position.y - current_radius,
            width: current_radius * 2.0,
            height: current_radius * 2.0,
        }
    }
    fn draw(&self, ctx: &mut Context) {
        let (width, height) = window::screen_size();
        let model = Mat4::from_scale_rotation_translation(
            vec3(DEFAULT_SCALE_FACTTOR_X, DEFAULT_SCALE_FACTTOR_Y, 1.0),
            Quat::IDENTITY,
            vec3(self.object.position.x, self.object.position.y, 0.0)
        );
        let proj = Mat4::orthographic_rh_gl(0.0, width, height, 0.0, -1.0, 1.0);
        let mvp = proj * model;
        let bindings = Bindings {
            vertex_buffers: vec![self.vertex_buffer.clone()],
            index_buffer: self.index_buffer.clone(),
            images: vec![],
        };
        ctx.apply_bindings(&bindings);
        ctx.apply_uniforms(UniformsSource::table(&mvp));
        ctx.draw(0, self.vertices.len() as i32, 1);
    }
    fn update(&mut self, surrounding_objects: Vec<SurroundingObject>) -> Option<GameEvent>{
        Updatable::update(self);
        let mut game_events = Vec::new();
        for object in surrounding_objects {
            if let Some(game_event) = self.check_and_handle_collision(&object.object, &object.hitbox) {
                game_events.push(game_event);
            }
        }
        return game_events.pop();
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
    fn get_hitbox(&self) -> Rect {
        Rect {
            x: self.object.position.x - 0.5,
            y: self.object.position.y - 0.5,
            width: 1.0,
            height: 1.0,
        }
    }
}

impl Bullet {
    fn new(ctx: &mut Context, pos: Vec2, velocity: Vec2) -> Bullet {
        let vertices = vec![
            Vec2::new(-0.1, 0.3),
            Vec2::new(0.1, 0.3),
            Vec2::new(0.1, -0.3),
            Vec2::new(-0.1, -0.3)
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
                BufferSource::slice(&vertex_buffer_data)
            ),
            index_buffer: ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&[0, 1, 2, 0, 2, 3])
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
        let normalized_velocity = self.velocity.normalize();
        let rotation = Quat::from_rotation_arc(vec3(0.0, -1.0, 0.0), vec3(normalized_velocity.x,normalized_velocity.y, 0.0));
        let model = Mat4::from_scale_rotation_translation(
            vec3(DEFAULT_SCALE_FACTTOR_X, DEFAULT_SCALE_FACTTOR_Y, 1.0),
            rotation,
            vec3(self.object.position.x, self.object.position.y, 0.0)
        );
        let proj = Mat4::orthographic_rh_gl(0.0, width, height, 0.0, -1.0, 1.0);
        let mvp = proj * model;
        let bindings = Bindings {
            vertex_buffers: vec![self.vertex_buffer.clone()],
            index_buffer: self.index_buffer.clone(),
            images: vec![],
        };
        ctx.apply_bindings(&bindings);
        ctx.apply_uniforms(UniformsSource::table(&mvp));
        ctx.draw(0, 6 as i32, 1);
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

    fn new() -> Self {
        let mut ctx = window::new_rendering_backend();
        let gfx_pipelines = Stage::load_shaders(&mut *ctx);
        let world = World::new(&mut *ctx, gfx_pipelines[0], gfx_pipelines[1]);
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
            Backend::OpenGl => {
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
    miniquad::start(
        Conf {
            window_title: "Rasteroid".to_owned(),
            window_width: WORLD_WIDTH as i32,
            window_height: WORLD_HEIGHT as i32,
            window_resizable: false,
            fullscreen: false,
            ..Default::default()
        },
        || Box::new(Stage::new())
    );
}
