#!/bin/bash
# Local development startup script (stub sampler only)

echo "🚀 Starting OMTRA webapp in LOCAL mode (stub sampler)"
echo "Environment: local"
echo "GPU: disabled"
echo "OMTRA model: disabled"

# Set environment variables
export ENVIRONMENT=local
export USE_GPU=false
export CUDA_VISIBLE_DEVICES=""
export OMTRA_MODEL_AVAILABLE=false
export API_URL=http://localhost:8000

# Start services
docker-compose up -d

echo ""
echo "✅ OMTRA webapp started!"
echo "🌐 Frontend: http://localhost:8501"
echo "🔧 API: http://localhost:8000"
echo "📊 Redis: localhost:6379"
echo ""
echo "📝 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"
