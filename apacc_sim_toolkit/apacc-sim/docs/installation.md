# APACC-Sim Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- Docker (for containerized deployment)

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://gitlab.com/apacc-sim/toolkit.git
cd apacc-sim

# Run with Docker Compose
docker-compose up -d

# Run validation
docker-compose run orchestrator python scripts/validate_controller.py \
    --controller examples/simple_controller.py \
    --monte-carlo 1000