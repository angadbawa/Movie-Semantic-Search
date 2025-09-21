from typing import Dict, Any, Optional
import yaml
from pathlib import Path
from functools import partial

class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get(self, key_path: Optional[str] = None) -> Any:
        """Get configuration value by dot-separated key path."""
        if key_path is None:
            return self._config
        
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Configuration key '{key_path}' not found")
        
        return value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._config.update(updates)
    
    def get_path(self, path_key: str, create_if_missing: bool = True) -> Path:
        """Get a path from configuration and optionally create it."""
        path_str = self.get(f"paths.{path_key}")
        path_obj = Path(path_str)
        
        if create_if_missing and not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)
        
        return path_obj

# Global configuration instance
_config = None

def get_config(key_path: Optional[str] = None) -> Any:
    """Get configuration value."""
    global _config
    if _config is None:
        _config = Config()
    return _config.get(key_path)

def update_config(updates: Dict[str, Any]) -> None:
    """Update global configuration."""
    global _config
    if _config is None:
        _config = Config()
    _config.update(updates)

def get_path(path_key: str, create_if_missing: bool = True) -> Path:
    """Get a configured path."""
    global _config
    if _config is None:
        _config = Config()
    return _config.get_path(path_key, create_if_missing)

# Convenience functions for common config sections
get_video_config = partial(get_config, "video")
get_detection_config = partial(get_config, "object_detection")
get_shot_config = partial(get_config, "shot_segmentation")
get_scene_config = partial(get_config, "scene_detection")
get_processing_config = partial(get_config, "processing")
