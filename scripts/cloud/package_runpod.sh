#!/bin/bash
#
# Package and Deploy RunPod Serverless Endpoint
#
# This script builds the Docker image for RunPod serverless predictions
# and pushes it to Docker Hub for deployment.
#
# Usage:
#   ./scripts/cloud/package_runpod.sh
#
# Prerequisites:
#   - Docker installed and running
#   - Docker Hub account
#   - DOCKER_USERNAME environment variable set
#   - Logged in to Docker Hub (docker login)
#
# Author: Statistical Modeling Agent
# Created: 2025-10-24 (Task 5.5: RunPod Deployment Script)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="ml-agent-prediction"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

# Check Docker Hub username
if [ -z "$DOCKER_USERNAME" ]; then
    echo -e "${YELLOW}⚠️  DOCKER_USERNAME not set${NC}"
    read -p "Enter your Docker Hub username: " DOCKER_USERNAME
    export DOCKER_USERNAME
fi

# Verify runpod directory exists
if [ ! -d "runpod" ]; then
    echo -e "${RED}❌ runpod/ directory not found${NC}"
    echo "Please run this script from the project root"
    exit 1
fi

# Build Docker image
echo ""
echo -e "${BLUE}🚀 Building RunPod serverless container...${NC}"
echo -e "${BLUE}   Platform: linux/amd64 (RunPod requirement)${NC}"
echo -e "${BLUE}   Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo ""

docker build \
    --platform linux/amd64 \
    -t "${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}" \
    -f runpod/Dockerfile \
    runpod/

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Container built successfully${NC}"
else
    echo -e "${RED}❌ Container build failed${NC}"
    exit 1
fi

# Push to Docker Hub
echo ""
echo -e "${BLUE}📤 Pushing to Docker Hub...${NC}"

docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image pushed successfully${NC}"
else
    echo -e "${RED}❌ Push failed${NC}"
    echo -e "${YELLOW}   Make sure you're logged in: docker login${NC}"
    exit 1
fi

# Display next steps
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ RunPod Deployment Package Ready!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Docker Image:${NC} ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "1. Go to RunPod Console: https://console.runpod.io → Serverless"
echo ""
echo "2. Create new serverless endpoint:"
echo "   - Name: ml-agent-prediction"
echo "   - Container Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "   - Select GPU type (e.g., NVIDIA RTX A5000)"
echo "   - Workers: Min 0, Max 3 (autoscaling)"
echo "   - Attach network volume for model storage"
echo ""
echo "3. Configure environment variables in RunPod:"
echo "   - STORAGE_ENDPOINT: https://storage.runpod.io"
echo "   - STORAGE_ACCESS_KEY: <your-storage-access-key>"
echo "   - STORAGE_SECRET_KEY: <your-storage-secret-key>"
echo ""
echo "4. Copy the endpoint ID and add to .env:"
echo "   RUNPOD_ENDPOINT_ID=<your-endpoint-id>"
echo ""
echo "5. Test the endpoint:"
echo "   python scripts/cloud/test_runpod_prediction.py"
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
