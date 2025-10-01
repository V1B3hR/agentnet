# AgentNet Docker Deployment Guide

This directory contains Docker and Docker Compose configurations for deploying AgentNet.

## Quick Start

### Using Docker

Build the image:
```bash
docker build -t agentnet:latest .
```

Run a container:
```bash
docker run -it --rm agentnet:latest python -m agentnet --demo sync
```

### Using Docker Compose

Start all services:
```bash
docker compose up -d
```

View logs:
```bash
docker compose logs -f agentnet
```

Stop all services:
```bash
docker compose down
```

## Services Included

The `docker-compose.yml` file includes the following services:

- **agentnet**: Main AgentNet application
- **postgres**: PostgreSQL 15 database for persistent storage
- **redis**: Redis for caching and rate limiting
- **prometheus**: Metrics collection and monitoring
- **grafana**: Dashboards and visualization (optional)

## Environment Configuration

The compose file uses the following default credentials:

**PostgreSQL:**
- User: `agentnet`
- Password: `agentnet_dev`
- Database: `agentnet_dev`

**Grafana:**
- Admin password: `admin`

> ⚠️ **Security Warning**: These are development defaults. Change them in production!

## Data Persistence

The following volumes are created for data persistence:

- `postgres-data`: PostgreSQL database files
- `redis-data`: Redis persistence
- `prometheus-data`: Prometheus metrics
- `grafana-data`: Grafana dashboards and configuration

Additionally, the following host directories are mounted:

- `./cost_logs`: Cost tracking data
- `./risk_data`: Risk register data
- `./mlops_data`: MLOps metadata
- `./configs`: Configuration files

## Accessing Services

Once services are running, you can access:

- **AgentNet API**: http://localhost:8000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## Customization

### Environment Variables

You can customize the AgentNet service by setting environment variables in `docker-compose.yml`:

```yaml
environment:
  - AGENTNET_ENV=production
  - LOG_LEVEL=WARNING
  - OPENAI_API_KEY=your-key-here
```

### Custom Configuration

Place your custom Prometheus configuration in `configs/prometheus.yml`.

## Production Deployment

For production use:

1. Change default passwords and credentials
2. Use Docker secrets for sensitive data
3. Configure SSL/TLS termination (nginx, traefik, etc.)
4. Set up proper backup strategies for volumes
5. Configure resource limits in the compose file
6. Use environment-specific compose files (e.g., `docker-compose.prod.yml`)

## Troubleshooting

### Container won't start
```bash
docker compose logs agentnet
```

### Database connection issues
```bash
docker compose exec postgres psql -U agentnet -d agentnet_dev
```

### Reset all data
```bash
docker compose down -v  # WARNING: This deletes all volumes!
```

## Building for Different Platforms

For multi-platform builds (e.g., ARM64 for Apple Silicon):

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t agentnet:latest .
```

## Health Checks

The AgentNet container includes a health check that verifies the package can be imported:

```bash
docker ps  # Check the STATUS column for health status
```

## Further Reading

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [AgentNet Documentation](../docs/)
