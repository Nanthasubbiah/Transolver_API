# Elasticity Solver API

FEM elasticity solver with ML (Transolver) warm start, exposed as a REST API.

## Quick Start

```bash
# 1. Place your solver code in core/ (the existing elasticity_api package)
# 2. Place .pt checkpoints in checkpoints/
# 3. Place .npy training data in train_data/

docker-compose up --build
```

API runs at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Usage

### Register & get API key
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret"}'

curl -X POST http://localhost:8000/auth/api-key \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret"}'
# → returns {"api_key": "elas_...", ...}
```

### List models
```bash
curl http://localhost:8000/models
```

### Solve
```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -H "X-API-Key: elas_your_key_here" \
  -d '{
    "nodes": [[0,0],[1,0],[0,1],...],
    "model_id": "linear_elastic_2d",
    "bc": {
      "symmetry": [
        {"axis": "x", "coord": 0.5},
        {"axis": "y", "coord": 0.5}
      ],
      "outer_traction": {"traction": 1.0}
    }
  }'
```

## Adding New Models

1. Place `.pt` file in `checkpoints/`
2. Place `.npy` normalizer data in `train_data/` (if data-loss model)
3. Add entry to `models.json`
4. Restart container

## Docker Hub (Private)

```bash
docker build -t yourusername/elasticity-api .
docker push yourusername/elasticity-api

# Friends pull:
docker pull yourusername/elasticity-api
docker run -p 8000:8000 yourusername/elasticity-api
```
