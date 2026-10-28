use shrinkage_widget::{Config, Method, scene, trace};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config: Config = if let Some(value) = std::env::args().nth(1) {
        serde_json::from_str(&value)?
    } else {
        Config::default()
    };
    let output = serde_json::json!({ "scene": scene(&config)?, "trace": trace(&config, Method::Proximal)?, "config": config });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
