## Multi stage build
### Before

```bash
docker build -t before_msb -f mlflow/Dockerfile --build-arg MLFLOW_VERSION=2.3.2 mlflow && docker run -p 5000:5000 before_msb
```

### After

```bash
docker build -t after_msb -f mlflow/Dockerfile-multistage-build --build-arg MLFLOW_VERSION=2.3.2 mlflow && docker run -p 5000:5000 after_msb
```

## Docker Compose

```bash
docker compose -f mlflow-docker-compose.yaml up -d
```

## Docker Swarm
```bash
# Add the current node to the swarm
docker swarm init
# Deploy a stack named trino
docker stack deploy -c postgresql-docker-compose.yaml trino
# Remove a stack named trino
docker stack rm trino
```