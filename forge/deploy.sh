#!/bin/bash

# Deploy Selector contracts using Forge scripts
# Usage: ./deploy.sh [rpc-url] [private-key] [owner]

# Default values
DEFAULT_RPC_URL="http://127.0.0.1:8545"
DEFAULT_PRIVATE_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# Set variables from command line arguments or use defaults
RPC_URL=${1:-$DEFAULT_RPC_URL}
PRIVATE_KEY=${2:-$DEFAULT_PRIVATE_KEY}

DEFAULT_OWNER=$(cast wallet address --private-key "$PRIVATE_KEY")
OWNER=${3:-$DEFAULT_OWNER}

# Set environment variable for selector factory owner
export SELECTOR_FACTORY_OWNER="$OWNER"

echo "=== Deploying Selector Contracts ==="
echo "RPC URL: $RPC_URL"
echo "Account: $(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Owner: $OWNER"
echo ""

# Check if forge is available
if ! command -v forge &> /dev/null; then
    echo "Error: forge command not found. Please install Foundry first."
    exit 1
fi

# Deploy SelectorFactory
echo "Step 1: Deploying SelectorFactory..."
forge script script/deployments/selectors/SelectorFactory.s.sol \
    --rpc-url "$RPC_URL" \
    --private-key "$PRIVATE_KEY" \
    --broadcast

if [ $? -eq 0 ]; then
    echo "✅ SelectorFactory deployed successfully"
else
    echo "❌ SelectorFactory deployment failed"
    exit 1
fi

echo ""

# Deploy AlwaysSampled
echo "Step 2: Deploying and registering AlwaysSampled..."
forge script script/deployments/selectors/AlwaysSampled.s.sol \
    --rpc-url "$RPC_URL" \
    --private-key "$PRIVATE_KEY" \
    --broadcast

if [ $? -eq 0 ]; then
    echo "✅ AlwaysSampled deployed and registered successfully"
else
    echo "❌ AlwaysSampled deployment failed"
    exit 1
fi

echo ""

# Deploy RandomSampling
echo "Step 3: Deploying and registering RandomSampling..."
forge script script/deployments/selectors/RandomSampling.s.sol \
    --rpc-url "$RPC_URL" \
    --private-key "$PRIVATE_KEY" \
    --broadcast

if [ $? -eq 0 ]; then
    echo "✅ RandomSampling deployed and registered successfully"
else
    echo "❌ RandomSampling deployment failed"
    exit 1
fi

echo ""

# Deploy SwarmV1Factory (depends on SelectorFactory)
echo "Step 4: Deploying SwarmV1Factory..."
forge script script/deployments/SwarmV1Factory.s.sol \
    --rpc-url "$RPC_URL" \
    --private-key "$PRIVATE_KEY" \
    --broadcast

if [ $? -eq 0 ]; then
    echo "✅ SwarmV1Factory deployed successfully"
else
    echo "❌ SwarmV1Factory deployment failed"
    exit 1
fi

echo ""
echo "=== All Contracts Deployed Successfully! ==="
echo "Check the broadcast logs for deployment addresses."


