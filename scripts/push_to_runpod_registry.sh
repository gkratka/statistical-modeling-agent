#!/bin/bash
#
# Push Docker image to RunPod Container Registry
# This script builds, tags, and pushes the ML training image to RunPod
#
# Usage:
#   ./scripts/push_to_runpod_registry.sh <username> <version>
#
# Example:
#   ./scripts/push_to_runpod_registry.sh johndoe v1.0.0
#
# Prerequisites:
#   - Docker installed and running
#   - RUNPOD_API_KEY environment variable set
#   - RunPod account with registry access
#
# Author: Statistical Modeling Agent
# Created: 2025-11-11

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check arguments
if [ $# -lt 1 ]; then
    log_error "Missing required arguments"
    echo "Usage: $0 <username> [version]"
    echo "Example: $0 johndoe v1.0.0"
    exit 1
fi

USERNAME="$1"
VERSION="${2:-latest}"
IMAGE_NAME="ml-training"
REGISTRY="registry.runpod.io"
FULL_IMAGE="${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION}"

log_info "RunPod Registry Push Script"
echo "----------------------------------------"
echo "Username:     ${USERNAME}"
echo "Image Name:   ${IMAGE_NAME}"
echo "Version:      ${VERSION}"
echo "Full Image:   ${FULL_IMAGE}"
echo "----------------------------------------"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if RUNPOD_API_KEY is set
if [ -z "${RUNPOD_API_KEY:-}" ]; then
    log_error "RUNPOD_API_KEY environment variable is not set"
    echo "Export your RunPod API key: export RUNPOD_API_KEY=your_key_here"
    exit 1
fi

# Check if Dockerfile exists
DOCKERFILE_PATH="docker/ml-training.Dockerfile"
if [ ! -f "${DOCKERFILE_PATH}" ]; then
    log_error "Dockerfile not found at: ${DOCKERFILE_PATH}"
    exit 1
fi

# Step 1: Build Docker image
log_info "Step 1/4: Building Docker image..."
docker build \
    --platform linux/amd64 \
    -f "${DOCKERFILE_PATH}" \
    -t "${IMAGE_NAME}:${VERSION}" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    .

if [ $? -ne 0 ]; then
    log_error "Docker build failed"
    exit 1
fi

log_info "Build successful: ${IMAGE_NAME}:${VERSION}"

# Step 2: Tag for RunPod registry
log_info "Step 2/4: Tagging image for RunPod registry..."
docker tag "${IMAGE_NAME}:${VERSION}" "${FULL_IMAGE}"

if [ $? -ne 0 ]; then
    log_error "Docker tag failed"
    exit 1
fi

log_info "Tagged as: ${FULL_IMAGE}"

# Step 3: Login to RunPod registry
log_info "Step 3/4: Logging in to RunPod registry..."
echo "${RUNPOD_API_KEY}" | docker login "${REGISTRY}" -u "${USERNAME}" --password-stdin

if [ $? -ne 0 ]; then
    log_error "Docker login failed. Check your RUNPOD_API_KEY and username."
    exit 1
fi

log_info "Login successful"

# Step 4: Push to registry
log_info "Step 4/4: Pushing image to RunPod registry..."
log_warn "This may take 5-10 minutes for first push (~8GB image)..."

docker push "${FULL_IMAGE}"

if [ $? -ne 0 ]; then
    log_error "Docker push failed"
    exit 1
fi

echo ""
log_info "✅ Successfully pushed to RunPod registry!"
echo ""
echo "Your image is now available at:"
echo "  ${FULL_IMAGE}"
echo ""
echo "Next steps:"
echo "  1. Update config/config.yaml:"
echo "     docker_image: \"${FULL_IMAGE}\""
echo ""
echo "  2. Update .env with registry credentials:"
echo "     RUNPOD_REGISTRY_USERNAME=${USERNAME}"
echo "     RUNPOD_REGISTRY_PASSWORD=\${RUNPOD_API_KEY}"
echo ""
echo "  3. Restart the bot:"
echo "     python3 src/bot/telegram_bot.py"
echo ""

# Cleanup (optional)
read -p "Remove local image tags? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rmi "${IMAGE_NAME}:${VERSION}" "${FULL_IMAGE}" || true
    log_info "Local images removed"
fi

log_info "Done!"
