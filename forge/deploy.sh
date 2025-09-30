#!/bin/bash

# Deploy All Federated Learning Contracts using Forge scripts
# Usage: ./deploy.sh [rpc-url] [private-key] [owner-address]

# Default values
DEFAULT_RPC_URL="http://127.0.0.1:8545"
DEFAULT_PRIVATE_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# Set variables from command line arguments or use defaults
RPC_URL=${1:-$DEFAULT_RPC_URL}
PRIVATE_KEY=${2:-$DEFAULT_PRIVATE_KEY}

DEFAULT_OWNER=$(cast wallet address --private-key "$PRIVATE_KEY")
OWNER=${3:-$DEFAULT_OWNER}

# Set environment variables for all factory owners
export SELECTOR_FACTORY_OWNER="$OWNER"
export CALCULATOR_FACTORY_OWNER="$OWNER"
export ACCESS_CONTROL_FACTORY_OWNER="$OWNER"
export COMPENSATION_FACTORY_OWNER="$OWNER"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function to deploy a contract
deploy_contract() {
    local script_path="$1"
    local contract_name="$2"
    local step_number="$3"
    
    echo -e "${BLUE}Step $step_number: Deploying $contract_name...${NC}"
    
    if forge script "$script_path" \
        --rpc-url "$RPC_URL" \
        --private-key "$PRIVATE_KEY" \
        --broadcast; then
        echo -e "${GREEN}✅ $contract_name deployed successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ $contract_name deployment failed${NC}"
        return 1
    fi
}

# Helper function to print section headers
print_section() {
    local section_name="$1"
    echo ""
    echo -e "${YELLOW}=== $section_name ===${NC}"
    echo ""
}

# Main deployment information
echo -e "${YELLOW}=== Federated Learning Contracts Deployment ===${NC}"
echo "RPC URL: $RPC_URL"
echo "Account: $(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Owner: $OWNER"
echo ""

# Check if forge is available
if ! command -v forge &> /dev/null; then
    echo -e "${RED}Error: forge command not found. Please install Foundry first.${NC}"
    exit 1
fi

# Check if cast is available
if ! command -v cast &> /dev/null; then
    echo -e "${RED}Error: cast command not found. Please install Foundry first.${NC}"
    exit 1
fi

# ============================================================================
# FACTORY DEPLOYMENTS
# ============================================================================

print_section "Deploying Factory Contracts"

# Deploy SelectorFactory
deploy_contract "script/deployments/selectors/SelectorFactory.s.sol" "SelectorFactory" "1" || exit 1

# Deploy CalculatorFactory
deploy_contract "script/deployments/calculators/CalculatorFactory.s.sol" "CalculatorFactory" "2" || exit 1

# Deploy AccessControlFactory
deploy_contract "script/deployments/access_control/AccessControlFactory.s.sol" "AccessControlFactory" "3" || exit 1

# Deploy CompensationFactory
deploy_contract "script/deployments/compensation/CompensationFactory.s.sol" "CompensationFactory" "4" || exit 1

# ============================================================================
# SELECTOR IMPLEMENTATIONS
# ============================================================================

print_section "Deploying Selector Implementations"

# Deploy AlwaysSampled
deploy_contract "script/deployments/selectors/AlwaysSampled.s.sol" "AlwaysSampled" "5" || exit 1

# Deploy RandomSampling
deploy_contract "script/deployments/selectors/RandomSampling.s.sol" "RandomSampling" "6" || exit 1

# ============================================================================
# CALCULATOR IMPLEMENTATIONS
# ============================================================================

print_section "Deploying Calculator Implementations"

# Deploy ContributionCalculator
deploy_contract "script/deployments/calculators/ContributionCalculator.s.sol" "ContributionCalculator" "7" || exit 1

# ============================================================================
# ACCESS CONTROL IMPLEMENTATIONS
# ============================================================================

print_section "Deploying Access Control Implementations"

# Deploy BaseAccessControl
deploy_contract "script/deployments/access_control/BaseAccessControl.s.sol" "BaseAccessControl" "8" || exit 1

# ============================================================================
# COMPENSATION IMPLEMENTATIONS
# ============================================================================

print_section "Deploying Compensation Implementations"

# Deploy SimpleMintCompensation
deploy_contract "script/deployments/compensation/SimpleMintCompensation.s.sol" "SimpleMintCompensation" "9" || exit 1

# ============================================================================
# MAIN SWARM FACTORY
# ============================================================================

print_section "Deploying Main Swarm Factory"

# Deploy SwarmV1Factory (depends on all other factories)
deploy_contract "script/deployments/SwarmV1Factory.s.sol" "SwarmV1Factory" "10" || exit 1

# ============================================================================
# DEPLOYMENT SUMMARY
# ============================================================================

echo ""
echo -e "${GREEN}=== 🎉 All Contracts Deployed Successfully! ===${NC}"
echo ""
echo -e "${BLUE}Deployment Summary:${NC}"
echo "├── Factory Contracts:"
echo "│   ├── SelectorFactory"
echo "│   ├── CalculatorFactory"
echo "│   ├── AccessControlFactory"
echo "│   └── CompensationFactory"
echo "├── Selector Implementations:"
echo "│   ├── AlwaysSampled"
echo "│   └── RandomSampling"
echo "├── Calculator Implementations:"
echo "│   └── ContributionCalculator"
echo "├── Access Control Implementations:"
echo "│   └── BaseAccessControl"
echo "├── Compensation Implementations:"
echo "│   └── SimpleMintCompensation"
echo "└── Main Factory:"
echo "    └── SwarmV1Factory"
echo ""
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo "1. Check the broadcast logs for deployment addresses"
echo "2. Update your configuration files with the deployed addresses"
echo "3. Verify all contracts are properly registered with their factories"
echo ""
echo -e "${BLUE}🔗 Useful Commands:${NC}"
echo "• View deployment logs: cat broadcast/*/run-latest.json"
echo "• Check factory registrations: forge script script/deployments/selectors/SelectorFactory.s.sol --rpc-url $RPC_URL"
echo "• Verify SwarmV1Factory: forge script script/deployments/SwarmV1Factory.s.sol --rpc-url $RPC_URL"
echo ""