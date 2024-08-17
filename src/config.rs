pub mod config {
    pub const WORLD_WIDTH: f32 = 512.0;
    pub const WORLD_HEIGHT: f32 = 512.0; 
    pub const PHYSICS_FRAME_LIMIT: f32 = 60.0;
    pub const PHYSICS_FRAME_TIME: f32 = 1.0 / PHYSICS_FRAME_LIMIT;
    pub const ASTEROID_BASE_SPEED: f32 = 15.0;
    pub const DEFAULT_SCALE_FACTTOR_X : f32 = WORLD_WIDTH / 12.0;
    pub const DEFAULT_SCALE_FACTTOR_Y : f32 = WORLD_HEIGHT / 12.0;
    pub const PLAYER_HITBOX_SIZE: f32 = 15.0;
    pub const TILE_SIZE: f32 = WORLD_WIDTH / 16.0;
}