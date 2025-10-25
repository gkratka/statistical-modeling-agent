#!/usr/bin/env python3
"""
RunPod Infrastructure Setup Script

This script automates the setup of RunPod resources for cloud ML workflows:
- Creates network volume for dataset and model storage
- Configures storage access keys
- Tests connectivity

Usage:
    python scripts/cloud/setup_runpod.py --config config/config.yaml
    python scripts/cloud/setup_runpod.py --config config/config.yaml --create-volume --volume-size 150

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 8.1: RunPod Setup Automation)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("❌ requests package is required. Install with: pip install requests")
    sys.exit(1)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.cloud.runpod_config import RunPodConfig
    from src.cloud.runpod_client import RunPodClient
except ImportError as e:
    print(f"❌ Failed to import RunPod modules: {e}")
    print("   Ensure you're running from project root and dependencies are installed")
    sys.exit(1)


def create_network_volume(
    api_key: str,
    name: str,
    size_gb: int,
    data_center_id: str = 'us-west'
) -> Optional[str]:
    """
    Create RunPod network volume using GraphQL API.

    Args:
        api_key: RunPod API key
        name: Volume name
        size_gb: Volume size in GB
        data_center_id: RunPod data center ID (default: 'us-west')

    Returns:
        Volume ID if successful, None otherwise

    Raises:
        requests.RequestException: If API request fails
    """
    print(f"Creating network volume: {name} ({size_gb}GB) in {data_center_id}...")

    query = """
    mutation CreateNetworkVolume($input: NetworkVolumeInput!) {
        createNetworkVolume(input: $input) {
            id
            name
            size
            dataCenterId
        }
    }
    """

    variables = {
        "input": {
            "name": name,
            "size": size_gb,
            "dataCenterId": data_center_id
        }
    }

    try:
        response = requests.post(
            'https://api.runpod.io/graphql',
            json={'query': query, 'variables': variables},
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=30
        )
        response.raise_for_status()

    except requests.RequestException as e:
        print(f"❌ Network request failed: {e}")
        return None

    result = response.json()

    if 'errors' in result:
        print(f"❌ Failed to create volume:")
        for error in result['errors']:
            print(f"   {error.get('message', error)}")
        return None

    if 'data' not in result or 'createNetworkVolume' not in result['data']:
        print(f"❌ Unexpected API response: {result}")
        return None

    volume_info = result['data']['createNetworkVolume']
    volume_id = volume_info['id']

    print(f"✅ Volume created successfully!")
    print(f"   Volume ID:     {volume_id}")
    print(f"   Volume Name:   {volume_info['name']}")
    print(f"   Size:          {volume_info['size']} GB")
    print(f"   Data Center:   {volume_info.get('dataCenterId', 'N/A')}")

    return volume_id


def test_connectivity(config: RunPodConfig) -> None:
    """
    Test RunPod API and storage connectivity.

    Args:
        config: RunPod configuration instance

    Prints health check results to stdout.
    """
    print("\n" + "="*60)
    print("Testing Connectivity")
    print("="*60)

    try:
        client = RunPodClient(config)
        health = client.health_check()

        # API connectivity
        if health.get('api'):
            print("✅ RunPod API: Connected")
            print(f"   Pod Count: {health.get('pod_count', 0)}")
        else:
            print("❌ RunPod API: Failed")
            if 'error' in health:
                print(f"   Error: {health['error']}")

        # Storage connectivity
        if health.get('storage'):
            print("✅ Storage endpoint: Accessible")
            if health.get('volume_accessible'):
                print(f"   Volume {config.network_volume_id}: Accessible")
            else:
                print(f"⚠️  Volume {config.network_volume_id}: Not accessible or empty")
        else:
            print("❌ Storage endpoint: Failed")

    except Exception as e:
        print(f"❌ Connectivity test failed: {e}")


def main() -> None:
    """Main entry point for RunPod setup script."""
    parser = argparse.ArgumentParser(
        description="Setup RunPod infrastructure for ML Agent cloud workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test connectivity with existing configuration
  python scripts/cloud/setup_runpod.py --config config/config.yaml

  # Create new network volume
  python scripts/cloud/setup_runpod.py --config config/config.yaml --create-volume

  # Create volume with custom size
  python scripts/cloud/setup_runpod.py --config config/config.yaml --create-volume --volume-size 150

Note:
  - RunPod API key must be configured in .env or config file
  - Network volume creation is optional (only needed for first-time setup)
  - Volume is billed at $0.07/GB/month for first 1TB
        """
    )
    parser.add_argument(
        '--config',
        required=True,
        help="Path to config.yaml file"
    )
    parser.add_argument(
        '--create-volume',
        action='store_true',
        help="Create new network volume"
    )
    parser.add_argument(
        '--volume-size',
        type=int,
        default=100,
        help="Volume size in GB (default: 100)"
    )
    parser.add_argument(
        '--data-center',
        default='us-west',
        help="RunPod data center ID (default: us-west)"
    )

    args = parser.parse_args()

    # Validate volume size
    if args.create_volume and args.volume_size < 1:
        print("❌ Volume size must be at least 1GB")
        sys.exit(1)

    # Load configuration
    print("="*60)
    print("RunPod Infrastructure Setup")
    print("="*60)
    print(f"\nLoading configuration from: {args.config}")

    try:
        config = RunPodConfig.from_yaml(args.config)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        print("\nEnsure your config file contains RunPod settings:")
        print("  runpod_api_key: your_api_key")
        print("  network_volume_id: your_volume_id  # (if already created)")
        sys.exit(1)

    # Validate API key
    if not config.runpod_api_key or not config.runpod_api_key.strip():
        print("❌ RunPod API key is missing")
        print("\nSet RUNPOD_API_KEY in .env or add to config file:")
        print("  runpod_api_key: your_api_key_here")
        sys.exit(1)

    # Create network volume if requested
    if args.create_volume:
        print("\n" + "="*60)
        print("Creating Network Volume")
        print("="*60)

        volume_id = create_network_volume(
            api_key=config.runpod_api_key,
            name="ml-agent-storage",
            size_gb=args.volume_size,
            data_center_id=args.data_center
        )

        if volume_id:
            print(f"\n⚠️  ACTION REQUIRED: Update your configuration with:")
            print(f"\n   In .env file:")
            print(f"   RUNPOD_NETWORK_VOLUME_ID={volume_id}")
            print(f"\n   Or in config/config.yaml:")
            print(f"   network_volume_id: {volume_id}")

            # Update config for connectivity test
            config.network_volume_id = volume_id
        else:
            print("\n❌ Volume creation failed. Skipping connectivity test.")
            sys.exit(1)

    # Test connectivity
    if config.network_volume_id:
        test_connectivity(config)
    else:
        print("\n⚠️  Skipping connectivity test (no volume ID configured)")
        print("   Run with --create-volume to create a new volume, or")
        print("   add existing volume ID to configuration")

    # Print next steps
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")

    if args.create_volume:
        print("1. ✅ Network volume created")
        print("2. ⚠️  Update .env with RUNPOD_NETWORK_VOLUME_ID (see above)")
        print("3. Configure storage access keys in .env:")
        print("   RUNPOD_STORAGE_ACCESS_KEY=your_access_key")
        print("   RUNPOD_STORAGE_SECRET_KEY=your_secret_key")
    else:
        print("1. Verify storage credentials are configured in .env")

    print("\n4. Build and push serverless prediction container:")
    print("   ./scripts/cloud/package_runpod.sh")
    print("\n5. Create serverless endpoint at console.runpod.io:")
    print("   - Go to Serverless → New Endpoint")
    print("   - Enter Docker image URL")
    print("   - Configure GPU and autoscaling")
    print("\n6. Update .env with RUNPOD_ENDPOINT_ID")
    print("\n7. Test the setup:")
    print("   See docs/runpod-testing-guide.md for testing instructions")
    print()


if __name__ == '__main__':
    main()
