// SPDX-License-Identifier: MIT
pragma solidity 0.8.25;

import {Test, console} from "forge-std/Test.sol";
import {SwarmV1} from "../../src/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "../../src/swarm/SwarmV1Factory.sol";
import {SelectorFactory} from "../../src/sampling/SelectorFactory.sol";
import {CalculatorFactory} from "../../src/contribution/CalculatorFactory.sol";
import {AccessControlFactory} from "../../src/access/AccessControlFactory.sol";
import {CompensationFactory} from "../../src/compensation/CompensationFactory.sol";

import {AlwaysSampled} from "../../src/sampling/AlwaysSampled.sol";
import {RandomSampling} from "../../src/sampling/RandomSampling.sol";
import {ContributionCalculator} from "../../src/contribution/ContributionCalculator.sol";
import {BaseAccessControl} from "../../src/access/BaseAccessControl.sol";
import {SimpleMintCompensation} from "../../src/compensation/SimpleMintCompensation.sol";
import {BaseTrainingPhases} from "../../src/training/BaseTrainingPhases.sol";

// ═══════════════════════════════════════════════════════════════════
//  Gas Profiling: sweep N=2..15, full claimReward pipeline per N
// ═══════════════════════════════════════════════════════════════════

contract SwarmV1GasProfileTest is Test {
    uint256 constant BLOCK_GAS_LIMIT = 60_000_000;
    uint256 constant NUM_EVALUATORS = 4;

    mapping(uint256 => bool) private _registered;

    // Shared factory infrastructure (deployed once in setUp)
    SwarmV1 implementation;
    SelectorFactory selectorFactory;
    CalculatorFactory calculatorFactory;
    AccessControlFactory accessControlFactory;
    CompensationFactory compensationFactory;
    bytes32 trainerSelectorId;
    bytes32 evaluatorSelectorId;
    bytes32 calculatorId;
    bytes32 accessControlId;
    bytes32 compensationId;

    function _realisticStratifiedAntitheticBudget(uint8 n) internal pure returns (uint256) {
        if (n == 8) return 256;
        if (n == 9) return 512;
        if (n == 10) return 972;
        if (n == 11) return 900;
        if (n == 12) return 798;
        if (n == 13) return 742;
        if (n == 14) return 674;
        if (n == 15) return 672;
        if (n == 16) return 598;
        revert("unsupported trainer count");
    }

    function setUp() public {
        // Deploy shared factory infrastructure once
        implementation = new SwarmV1();
        selectorFactory = new SelectorFactory(address(this));
        calculatorFactory = new CalculatorFactory(address(this));
        accessControlFactory = new AccessControlFactory(address(this));
        compensationFactory = new CompensationFactory(address(this));

        AlwaysSampled trainerSelectorImpl = new AlwaysSampled();
        RandomSampling evaluatorSelectorImpl = new RandomSampling();
        ContributionCalculator calculatorImpl = new ContributionCalculator();
        BaseAccessControl accessControlImpl = new BaseAccessControl();
        SimpleMintCompensation compensationImpl = new SimpleMintCompensation();

        (,, string memory v1,,,,) = trainerSelectorImpl.eip712Domain();
        (,, string memory v2,,,,) = evaluatorSelectorImpl.eip712Domain();
        (,, string memory v3,,,,) = calculatorImpl.eip712Domain();
        (,, string memory v4,,,,) = accessControlImpl.eip712Domain();
        (,, string memory v5,,,,) = compensationImpl.eip712Domain();

        trainerSelectorId = selectorFactory.getID(v1);
        evaluatorSelectorId = selectorFactory.getID(v2);
        calculatorId = calculatorFactory.getID(v3);
        accessControlId = accessControlFactory.getID(v4);
        compensationId = compensationFactory.getID(v5);

        selectorFactory.registerSelectorImplementation(address(trainerSelectorImpl));
        selectorFactory.registerSelectorImplementation(address(evaluatorSelectorImpl));
        calculatorFactory.registerCalculatorImplementation(address(calculatorImpl));
        accessControlFactory.registerAccessControlImplementation(address(accessControlImpl));
        compensationFactory.registerCompensationImplementation(address(compensationImpl));
    }

    /// @dev Deploy a fresh SwarmV1 with N trainers, numSamples = 2^N,
    ///      run the full round, and return the swarm + trainer addresses.
    function _deployAndRunRound(uint8 n, uint256 numSamples)
        internal
        returns (SwarmV1 swarm, address[] memory trainerAddrs)
    {
        address aggregator = makeAddr("aggregator");

        // Create trainer and evaluator addresses
        trainerAddrs = new address[](n);
        for (uint256 i = 0; i < n; i++) {
            trainerAddrs[i] = makeAddr(string(abi.encodePacked("t", vm.toString(n), "_", vm.toString(i))));
        }
        address[] memory evalAddrs = new address[](NUM_EVALUATORS);
        for (uint256 i = 0; i < NUM_EVALUATORS; i++) {
            evalAddrs[i] = makeAddr(string(abi.encodePacked("e", vm.toString(n), "_", vm.toString(i))));
        }

        // Deploy swarm
        SwarmV1Factory factory = new SwarmV1Factory(
            address(implementation),
            address(selectorFactory),
            address(calculatorFactory),
            address(accessControlFactory),
            address(compensationFactory)
        );

        bytes32 salt = keccak256(abi.encodePacked("gas-sweep-", n));
        address swarmAddress = factory.getSwarmAddress(salt);

        SwarmV1Factory.SwarmParams memory params = SwarmV1Factory.SwarmParams({
            swarm: SwarmV1Factory.SwarmV1Params({name: "GasSweep"}),
            trainerSelector: SwarmV1Factory.SelectorParams({
                id: trainerSelectorId,
                initData: abi.encodeWithSelector(AlwaysSampled.initialize.selector)
            }),
            evaluatorSelector: SwarmV1Factory.SelectorParams({
                id: evaluatorSelectorId,
                initData: abi.encodeWithSelector(RandomSampling.initialize.selector, 1 ether)
            }),
            contributionCalculator: SwarmV1Factory.CalculatorParams({
                id: calculatorId,
                initData: abi.encodeWithSelector(ContributionCalculator.initialize.selector, swarmAddress, numSamples)
            }),
            accessControl: SwarmV1Factory.AccessControlParams({
                id: accessControlId,
                initData: abi.encodeWithSelector(BaseAccessControl.initialize.selector, aggregator, trainerAddrs, evalAddrs)
            }),
            compensation: SwarmV1Factory.CompensationParams({
                id: compensationId,
                initData: abi.encodeWithSelector(
                    SimpleMintCompensation.initialize.selector, "GT", "GT", 100000 ether, aggregator, swarmAddress
                )
            }),
            trainingPhaseConfiguration: BaseTrainingPhases.TrainingPhaseConfiguration({ttl: 1000}),
            evaluationPhaseConfiguration: BaseTrainingPhases.EvaluationPhaseConfiguration({
                ttl: 100000,
                registrationTtl: 1000
            })
        });

        swarm = SwarmV1(factory.createSwarm(salt, params));
        ContributionCalculator calc = ContributionCalculator(swarm.getContributionCalculator());

        // ── Run full round ──
        uint256 roundId = 1;

        vm.prank(aggregator);
        swarm.startTrainingRound();

        for (uint256 i = 0; i < n; i++) {
            vm.prank(trainerAddrs[i]);
            swarm.registerRoundContribution(roundId, keccak256(abi.encodePacked("model", i)));
        }

        BaseTrainingPhases.TrainingPhaseConfiguration memory tCfg = swarm.getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + tCfg.ttl);
        swarm.updatePhase();

        for (uint256 i = 0; i < NUM_EVALUATORS; i++) {
            vm.prank(evalAddrs[i]);
            swarm.registerForRoundEvaluation(roundId);
        }

        BaseTrainingPhases.EvaluationPhaseConfiguration memory eCfg = swarm.getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + eCfg.registrationTtl);
        swarm.updatePhase();

        // Submit evaluations
        uint8 nTrainers = uint8(swarm.getTrainerCount(roundId));
        for (uint256 evalIdx = 0; evalIdx < NUM_EVALUATORS; evalIdx++) {
            vm.startPrank(evalAddrs[evalIdx]);
            uint256 evalId = swarm.getEvaluatorId(roundId, evalAddrs[evalIdx]);

            for (uint256 taskOffset = 0; taskOffset < numSamples; taskOffset++) {
                try swarm.nthTaskOfNode(roundId, evalId - 1, taskOffset) returns (uint256 taskId) {
                    uint256 targetMask = calc.getMask(roundId, taskId, nTrainers);

                    swarm.registerEvaluation(
                        roundId, taskId, targetMask, keccak256(abi.encodePacked("m", taskId)), int256(targetMask * 100)
                    );

                    for (uint8 bit = 0; bit < nTrainers; bit++) {
                        uint256 neighborMask = targetMask ^ (1 << bit);
                        if (!_registered[neighborMask]) {
                            swarm.registerEvaluation(
                                roundId,
                                taskId,
                                neighborMask,
                                keccak256(abi.encodePacked("m", taskId)),
                                int256(neighborMask * 100)
                            );
                            _registered[neighborMask] = true;
                        }
                    }
                    _registered[targetMask] = true;
                } catch {
                    break;
                }
            }
            vm.stopPrank();
        }

        // Clear registered mapping for next iteration
        for (uint256 m = 0; m < (uint256(1) << n); m++) {
            _registered[m] = false;
        }

        vm.warp(block.timestamp + eCfg.ttl);
        swarm.updatePhase();
    }

    /// @notice Sweep N=2..15: deploy fresh swarm, run full round,
    ///         measure claimReward gas for trainer 0 under ssk-last3 with deterministic backfill.
    function profile_claimReward_gasSweep_2to15() public {
        console.log("================================================================");
        console.log("  SSK-last3 Gas Sweep: N=2..15");
        console.log("  requestedSamples = min(2^N, 6000)");
        console.log("  Block gas limit: 60,000,000");
        console.log("================================================================");

        for (uint8 n = 2; n <= 15; n++) {
            uint256 coalitions = 1 << n;
            if (coalitions > 6000) coalitions = 6000;

            // Adjust deploy to use specific sample size
            (SwarmV1 swarm, address[] memory trainerAddrs) = _deployAndRunRound(n, coalitions);

            // Measure claimReward for trainer 0
            uint256 gasBefore = gasleft();
            swarm.claimReward(1, trainerAddrs[0]);
            uint256 gasUsed = gasBefore - gasleft();

            bool fits = gasUsed < BLOCK_GAS_LIMIT;

            console.log("---");
            console.log("N =", uint256(n));
            console.log("  trainers:", uint256(n));
            console.log("  measuredSamples:", coalitions);
            console.log("  gasUsed:", gasUsed);
            console.log("  perSample:", gasUsed / coalitions);
            console.log("  fitsIn60M:", fits);
            if (fits) {
                console.log("  headroom:", BLOCK_GAS_LIMIT - gasUsed);
                console.log("  budgetUsed%:", (gasUsed * 100) / BLOCK_GAS_LIMIT);
            } else {
                console.log("  overBudgetBy:", gasUsed - BLOCK_GAS_LIMIT);
            }
        }
    }

    /// @notice Sweep N=9..16 using the current stratified-antithetic emitted
    ///         sample ceilings from examples/torch_trainer_scaling/runall.py,
    ///         and measure end-to-end claimReward gas across all trainers.
    function profile_claimReward_gasSweep_realistic_9to16() public {
        console.log("================================================================");
        console.log("  claimReward Gas Sweep: realistic stratified-antithetic budgets");
        console.log("  Trainers: 9..16");
        console.log("  Block gas limit: 60,000,000");
        console.log("================================================================");

        for (uint8 n = 9; n <= 16; n++) {
            uint256 samples = _realisticStratifiedAntitheticBudget(n);
            (SwarmV1 swarm, address[] memory trainerAddrs) = _deployAndRunRound(n, samples);

            uint256 totalGas = 0;
            uint256 maxGas = 0;
            uint256 maxTrainerIndex = 0;

            for (uint256 trainerIdx = 0; trainerIdx < trainerAddrs.length; trainerIdx++) {
                uint256 gasBefore = gasleft();
                swarm.claimReward(1, trainerAddrs[trainerIdx]);
                uint256 gasUsed = gasBefore - gasleft();

                totalGas += gasUsed;
                if (gasUsed > maxGas) {
                    maxGas = gasUsed;
                    maxTrainerIndex = trainerIdx;
                }
            }

            uint256 avgGas = totalGas / trainerAddrs.length;

            console.log("---");
            console.log("N =", uint256(n));
            console.log("  emittedSamples:", samples);
            console.log("  avgClaimRewardGas:", avgGas);
            console.log("  maxClaimRewardGas:", maxGas);
            console.log("  maxTrainerIndex:", maxTrainerIndex);
            console.log("  avgFitsIn60M:", avgGas < BLOCK_GAS_LIMIT);
            console.log("  maxFitsIn60M:", maxGas < BLOCK_GAS_LIMIT);
            if (maxGas < BLOCK_GAS_LIMIT) {
                console.log("  maxHeadroom:", BLOCK_GAS_LIMIT - maxGas);
            } else {
                console.log("  maxOverBudgetBy:", maxGas - BLOCK_GAS_LIMIT);
            }
        }
    }

    function _logRealisticClaimRewardProfile(uint8 n) internal {
        uint256 samples = _realisticStratifiedAntitheticBudget(n);
        (SwarmV1 swarm, address[] memory trainerAddrs) = _deployAndRunRound(n, samples);
        uint256 trainerIdx = 0;
        uint256 gasBefore = gasleft();
        swarm.claimReward(1, trainerAddrs[trainerIdx]);
        uint256 gasUsed = gasBefore - gasleft();

        console.log("================================================================");
        console.log("  realistic claimReward gas");
        console.log("================================================================");
        console.log("N =", uint256(n));
        console.log("  emittedSamples:", samples);
        console.log("  trainerIndex:", trainerIdx);
        console.log("  claimRewardGas:", gasUsed);
        console.log("  fitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);
        if (gasUsed < BLOCK_GAS_LIMIT) {
            console.log("  headroom:", BLOCK_GAS_LIMIT - gasUsed);
        } else {
            console.log("  overBudgetBy:", gasUsed - BLOCK_GAS_LIMIT);
        }
    }

    function _measureClaimRewardGas(uint8 n, uint256 emittedSamples) internal returns (uint256 gasUsed) {
        (SwarmV1 swarm, address[] memory trainerAddrs) = _deployAndRunRound(n, emittedSamples);
        uint256 gasBefore = gasleft();
        swarm.claimReward(1, trainerAddrs[0]);
        gasUsed = gasBefore - gasleft();
    }

    function _findMaxRealisticClaimRewardSamples(uint8 n) internal returns (uint256 bestSamples, uint256 bestGas) {
        uint256 maxSamples = _realisticStratifiedAntitheticBudget(n);
        uint256 lowPairs = 1;
        uint256 highPairs = maxSamples / 2;

        while (lowPairs <= highPairs) {
            uint256 midPairs = (lowPairs + highPairs) / 2;
            uint256 samples = midPairs * 2;
            uint256 gasUsed = _measureClaimRewardGas(n, samples);

            console.log("  probeSamples:", samples);
            console.log("  probeGas:", gasUsed);
            console.log("  probeFitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);

            if (gasUsed < BLOCK_GAS_LIMIT) {
                bestSamples = samples;
                bestGas = gasUsed;
                lowPairs = midPairs + 1;
            } else {
                highPairs = midPairs - 1;
            }
        }
    }

    function _logMaxRealisticClaimRewardSamples(uint8 n) internal {
        uint256 startingSamples = _realisticStratifiedAntitheticBudget(n);
        (uint256 bestSamples, uint256 bestGas) = _findMaxRealisticClaimRewardSamples(n);

        console.log("================================================================");
        console.log("  max realistic claimReward samples under 60M");
        console.log("================================================================");
        console.log("N =", uint256(n));
        console.log("  startingSamples:", startingSamples);
        console.log("  maxSamplesUnder60M:", bestSamples);
        console.log("  claimRewardGasAtMax:", bestGas);
        console.log("  reduction:", startingSamples - bestSamples);
        console.log("  coverageOfStarting%:", (bestSamples * 100) / startingSamples);
        console.log("  headroom:", BLOCK_GAS_LIMIT - bestGas);
    }

    function test_claimReward_gas_realistic_n9() public {
        _logRealisticClaimRewardProfile(9);
    }

    function test_claimReward_gas_realistic_n10() public {
        _logRealisticClaimRewardProfile(10);
    }

    function test_claimReward_gas_realistic_n11() public {
        _logRealisticClaimRewardProfile(11);
    }

    function test_claimReward_gas_realistic_n12() public {
        _logRealisticClaimRewardProfile(12);
    }

    function test_claimReward_gas_realistic_n13() public {
        _logRealisticClaimRewardProfile(13);
    }

    function test_claimReward_gas_realistic_n14() public {
        _logRealisticClaimRewardProfile(14);
    }

    function test_claimReward_gas_realistic_n15() public {
        _logRealisticClaimRewardProfile(15);
    }

    function test_claimReward_gas_realistic_n16() public {
        _logRealisticClaimRewardProfile(16);
    }

    function test_claimReward_findMaxRealisticSamples_n9() public {
        _logMaxRealisticClaimRewardSamples(9);
    }

    function test_claimReward_findMaxRealisticSamples_n10() public {
        _logMaxRealisticClaimRewardSamples(10);
    }

    function test_claimReward_findMaxRealisticSamples_n11() public {
        _logMaxRealisticClaimRewardSamples(11);
    }

    function test_claimReward_findMaxRealisticSamples_n12() public {
        _logMaxRealisticClaimRewardSamples(12);
    }

    function test_claimReward_findMaxRealisticSamples_n13() public {
        _logMaxRealisticClaimRewardSamples(13);
    }

    function test_claimReward_findMaxRealisticSamples_n14() public {
        _logMaxRealisticClaimRewardSamples(14);
    }

    function test_claimReward_findMaxRealisticSamples_n15() public {
        _logMaxRealisticClaimRewardSamples(15);
    }

    function test_claimReward_findMaxRealisticSamples_n16() public {
        _logMaxRealisticClaimRewardSamples(16);
    }

    /// @notice Binary search: exact max numSamples at N=13 within 60M claimReward
    function test_claimReward_findMaxSamples_sskLast3() public {
        uint8 n = 13;
        uint256 totalAvailableSamples = 1 << n;
        (SwarmV1 swarm,) = _deployAndRunRound(n, totalAvailableSamples);
        ContributionCalculator calc = ContributionCalculator(swarm.getContributionCalculator());

        uint256 lo = 1;
        uint256 hi = calc.getTotalEvaluations(1, n);
        uint256 best = 0;
        uint256 bestGas = 0;

        console.log("================================================================");
        console.log("  Binary Search (SSK-last3 + backfill): max numSamples at N=13");
        console.log("================================================================");

        while (lo <= hi) {
            uint256 mid = (lo + hi) / 2;

            vm.prank(address(swarm));
            calc.setEvaluationsRequired(1, mid);

            uint256 g0 = gasleft();
            calc.calculateContribution(1, 0, n);
            uint256 gasUsed = g0 - gasleft();

            // Add ~62K for SwarmV1 claimReward overhead
            uint256 totalEstimated = gasUsed + 62000;

            if (totalEstimated < BLOCK_GAS_LIMIT) {
                best = mid;
                bestGas = totalEstimated;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        console.log("SSK-last3 Max numSamples:", best);
        console.log("Estimated claimReward gas:", bestGas);
        console.log("Headroom:", BLOCK_GAS_LIMIT - bestGas);
    }
}
