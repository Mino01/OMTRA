import os
from typing import Optional

class Config:
    """Environment-based configuration for OMTRA webapp"""
    
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
    
    # API URLs based on environment
    if ENVIRONMENT == 'local':
        API_URL = "http://localhost:8000"
        REDIS_URL = os.getenv('REDIS_URL', "redis://localhost:6379")
    elif ENVIRONMENT == 'gcp':
        API_URL = "http://api:8000"  # Docker internal
        REDIS_URL = os.getenv('REDIS_URL', "redis://redis:6379")
    elif ENVIRONMENT == 'production':
        API_URL = "http://localhost:8000"  # Same server
        REDIS_URL = os.getenv('REDIS_URL', "redis://localhost:6379")
    else:
        # Default to local
        API_URL = "http://localhost:8000"
        REDIS_URL = os.getenv('REDIS_URL', "redis://localhost:6379")
    
    # GPU settings
    USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0' if USE_GPU else '')
    
    # Model paths
    MODEL_CHECKPOINT_PATH = os.getenv('MODEL_CHECKPOINT_PATH', '/srv/app/models/checkpoint.ckpt')
    DISTS_FILE = os.getenv('DISTS_FILE', '/srv/app/models/train_dists.npz')
    PHARMIT_PATH = os.getenv('PHARMIT_PATH', '/srv/app/data/pharmit')  # optional
    
    # OMTRA model settings
    OMTRA_MODEL_AVAILABLE = os.getenv('OMTRA_MODEL_AVAILABLE', 'false').lower() == 'true'
    
    @classmethod
    def get_api_url(cls) -> str:
        """Get API URL for current environment"""
        return cls.API_URL
    
    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis URL for current environment"""
        return cls.REDIS_URL
    
    @classmethod
    def is_gpu_enabled(cls) -> bool:
        """Check if GPU is enabled for current environment"""
        return cls.USE_GPU
    
    @classmethod
    def is_omtra_model_available(cls) -> bool:
        """Check if OMTRA model is available"""
        return cls.OMTRA_MODEL_AVAILABLE
    
    @classmethod
    def get_model_path(cls) -> str:
        """Get model checkpoint path"""
        return cls.MODEL_CHECKPOINT_PATH
    
    @classmethod
    def get_dists_file(cls) -> str:
        """Get distribution file path"""
        return cls.DISTS_FILE
    
    @classmethod
    def get_pharmit_path(cls) -> str:
        """Get Pharmit dataset path"""
        return cls.PHARMIT_PATH
