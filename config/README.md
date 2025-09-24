# Configuration Files

This directory contains configuration files for the Movie Semantic Search system.

## Files

### default.yaml
The main configuration file containing all system settings including:
- Video processing parameters
- Model configurations
- Storage paths
- Database settings

### example_actors.json
Example actor configuration file showing how to set up actor recognition database.

## Usage

### Basic Configuration
The system will automatically load `default.yaml` for configuration. You can override settings by:

1. Modifying `default.yaml` directly
2. Creating a custom config file and passing it via `--config-file` argument

### Actor Database Setup
To use actor recognition features:

1. Copy `example_actors.json` to a new file (e.g., `my_actors.json`)
2. Update the actor information with real image paths or pre-computed encodings
3. Initialize the database using:
   ```bash
   python main.py multimodal init-actors --config config/my_actors.json
   ```

### Actor Configuration Format
```json
{
  "actor_name": {
    "image_paths": ["path/to/image1.jpg", "path/to/image2.jpg"],
    "metadata": {
      "full_name": "Actor Full Name",
      "age": 30,
      "gender": "male/female"
    }
  }
}
```

Alternatively, use pre-computed encodings:
```json
{
  "actor_name": {
    "encodings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "metadata": {...}
  }
}
```
