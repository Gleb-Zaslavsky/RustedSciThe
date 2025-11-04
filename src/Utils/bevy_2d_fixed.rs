use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use nalgebra::{DMatrix, DVector};

#[derive(Resource)]
struct PlotData {
    x: DVector<f64>,
    ys: DMatrix<f64>,
    names: Vec<String>,
}

#[derive(Component)]
struct CameraController {
    zoom: f32,
    offset: Vec2,
}

pub fn plot_static_functions(x: DVector<f64>, ys: DMatrix<f64>, names: Vec<String>) {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(ClearColor(Color::srgb(0.97, 0.97, 0.99)))
        .insert_resource(PlotData { x, ys, names })
        .add_systems(Startup, setup_plot_scene)
        .add_systems(Update, (draw_static_plot, camera_2d_controller))
        .run();
}

fn setup_plot_scene(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)).with_scale(Vec3::splat(0.01)),
        CameraController {
            offset: Vec2::ZERO,
            zoom: 0.01,
        },
    ));
}

fn draw_static_plot(mut gizmos: Gizmos, plot: Res<PlotData>) {
    let x = &plot.x;
    let ys = &plot.ys;

    if ys.ncols() != x.len() {
        return;
    }

    let colors = [
        Color::srgb(1.0, 0.0, 0.0),
        Color::srgb(0.0, 0.7, 0.0),
        Color::srgb(0.0, 0.3, 1.0),
        Color::srgb(1.0, 0.5, 0.0),
        Color::srgb(0.7, 0.0, 0.7),
    ];

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for i in 0..x.len() {
        let xi = x[i];
        min_x = min_x.min(xi);
        max_x = max_x.max(xi);
        for r in 0..ys.nrows() {
            let yv = ys[(r, i)];
            min_y = min_y.min(yv);
            max_y = max_y.max(yv);
        }
    }

    let xrange = (max_x - min_x).max(1e-6);
    let yrange = (max_y - min_y).max(1e-6);
    let scale_factor = 500.0;

    // Draw axes
    let x_center = (min_x + max_x) * 0.5;
    let y_center = (min_y + max_y) * 0.5;
    
    gizmos.line_2d(
        Vec2::new(((min_x - 0.1 * xrange) * scale_factor) as f32, (y_center * scale_factor) as f32),
        Vec2::new(((max_x + 0.1 * xrange) * scale_factor) as f32, (y_center * scale_factor) as f32),
        Color::srgb(0.7, 0.7, 0.7)
    );
    gizmos.line_2d(
        Vec2::new((x_center * scale_factor) as f32, ((min_y - 0.1 * yrange) * scale_factor) as f32),
        Vec2::new((x_center * scale_factor) as f32, ((max_y + 0.1 * yrange) * scale_factor) as f32),
        Color::srgb(0.7, 0.7, 0.7)
    );

    // Draw curves
    for (r, name) in plot.names.iter().enumerate() {
        let color = colors[r % colors.len()];

        for i in 0..x.len() - 1 {
            let p1 = Vec2::new((x[i] * scale_factor) as f32, (ys[(r, i)] * scale_factor) as f32);
            let p2 = Vec2::new((x[i + 1] * scale_factor) as f32, (ys[(r, i + 1)] * scale_factor) as f32);
            gizmos.line_2d(p1, p2, color);
        }

        let last = Vec2::new((x[x.len() - 1] * scale_factor) as f32, (ys[(r, x.len() - 1)] * scale_factor) as f32);
        gizmos.circle_2d(last, 5.0, color);
    }
}

fn camera_2d_controller(
    mut mouse_motion: EventReader<MouseMotion>,
    mut scroll_events: EventReader<MouseWheel>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut camera_query: Query<(&mut Transform, &mut CameraController), With<Camera2d>>,
) {
    for (mut transform, mut controller) in camera_query.iter_mut() {
        // Pan
        if mouse_input.pressed(MouseButton::Left) {
            for motion in mouse_motion.read() {
                let pan_speed = 3.0 / controller.zoom;
                controller.offset.x -= motion.delta.x * pan_speed;
                controller.offset.y += motion.delta.y * pan_speed;
            }
        }

        // Zoom
        for scroll in scroll_events.read() {
            let zoom_factor = 1.0 + scroll.y * 0.15;
            controller.zoom *= zoom_factor;
            controller.zoom = controller.zoom.clamp(0.001, 2.0);
        }

        transform.translation.x = controller.offset.x;
        transform.translation.y = controller.offset.y;
        transform.scale = Vec3::splat(controller.zoom);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_2d() {
        let n = 200;
        let x = DVector::from_iterator(n, (0..n).map(|i| i as f64 * 0.05));
        let mut ys = DMatrix::zeros(3, n);
        for i in 0..n {
            let xi = x[i];
            ys[(0, i)] = xi.sin();
            ys[(1, i)] = xi.cos();
            ys[(2, i)] = 0.5 * (2.0 * xi).sin();
        }

        let names = vec!["sin(x)".into(), "cos(x)".into(), "0.5*sin(2x)".into()];
        plot_static_functions(x, ys, names);
    }
}