// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console} from "forge-std/Test.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {ContributionCalculator} from "../../src/contribution/ContributionCalculator.sol";
import {SwarmV1} from "../../src/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "../../src/swarm/SwarmV1Factory.sol";
import {SelectorFactory} from "../../src/sampling/SelectorFactory.sol";
import {CalculatorFactory} from "../../src/contribution/CalculatorFactory.sol";
import {AlwaysSampled} from "../../src/sampling/AlwaysSampled.sol";
import {RandomSampling} from "../../src/sampling/RandomSampling.sol";
import {BaseTrainingPhases} from "../../src/training/BaseTrainingPhases.sol";
import {BaseAccessControl} from "../../src/access/BaseAccessControl.sol";
import {SimpleMintCompensation} from "../../src/compensation/SimpleMintCompensation.sol";
import {AccessControlFactory} from "../../src/access/AccessControlFactory.sol";
import {CompensationFactory} from "../../src/compensation/CompensationFactory.sol";

/**
 * @title ShapleyGasBudgetTest
 * @notice Complete gas budget simulation for _calcShapley / claimReward
 *         using the real deterministic RandPerm (Feistel) mask generator.
 *
 *         NUM_TRAINERS = 13  -->  2^13 = 8192 total coalitions.
 *
 *         Tests:
 *          1. Sweep numSamples 10..8192 and log gas for calculateContribution
 *          2. Binary search for the exact max numSamples within 60M gas
 *          3. Full claimReward simulation through SwarmV1 for all trainers
 *          4. Per-sample gas cost extrapolation to 8..20 trainers
 */

// ═══════════════════════════════════════════════════════════════════
//  Part 1: ContributionCalculator-level gas profiling
// ═══════════════════════════════════════════════════════════════════

contract ShapleyGasBudgetTest is Test {
    ContributionCalculator public calculator;
    address public admin;
    mapping(uint256 => bool) private _registered;

    uint8 constant NUM_TRAINERS = 13;
    uint256 constant ROUND_ID = 1;
    uint256 constant TOTAL_COALITIONS = 1 << NUM_TRAINERS; // 8192
    uint256 constant BLOCK_GAS_LIMIT = 60_000_000;

    function setUp() public {
        admin = makeAddr("admin");

        // Deploy ContributionCalculator with real RandPerm _getMask
        ContributionCalculator impl = new ContributionCalculator();
        bytes memory initData =
            abi.encodeWithSelector(ContributionCalculator.initialize.selector, admin, TOTAL_COALITIONS);
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        calculator = ContributionCalculator(address(proxy));

        // Register evaluation results for ALL 8192 coalition masks
        vm.prank(admin);
        calculator.setEvaluationsRequired(ROUND_ID, TOTAL_COALITIONS);
        _registerAllCoalitions();
    }

    /// @dev For each sampleId 0..8191, get the deterministic target mask,
    ///      then register it (distance=0) and all its 1-bit neighbors (distance=1).
    ///      This ensures every possible coalition mask has a stored result.
    function _registerAllCoalitions() internal {
        vm.startPrank(admin);
        bytes32 modelHash = keccak256("gas_test_model");

        for (uint256 sampleId = 0; sampleId < TOTAL_COALITIONS; sampleId++) {
            uint256 targetMask = calculator.getMask(ROUND_ID, sampleId, NUM_TRAINERS);

            // Register target mask (distance = 0)
            if (!_registered[targetMask]) {
                calculator.registerResult(
                    ROUND_ID, sampleId, targetMask, modelHash, int256(targetMask * 100), NUM_TRAINERS
                );
                _registered[targetMask] = true;
            }

            // Register each 1-bit-flip neighbor (distance = 1)
            for (uint8 bit = 0; bit < NUM_TRAINERS; bit++) {
                uint256 neighborMask = targetMask ^ (1 << bit);
                if (!_registered[neighborMask]) {
                    calculator.registerResult(
                        ROUND_ID, sampleId, neighborMask, modelHash, int256(neighborMask * 100), NUM_TRAINERS
                    );
                    _registered[neighborMask] = true;
                }
            }
        }
        vm.stopPrank();
    }

    // ─────────────────────────────────────────────────────────────
    //  Test 1: Sweep numSamples and measure calculateContribution gas
    // ─────────────────────────────────────────────────────────────

    function test_gasProfile_calcShapley_sweep() public {
        uint256 trainerIndex = 0;

        uint256[20] memory sampleCounts = [
            uint256(10),
            uint256(50),
            uint256(100),
            uint256(200),
            uint256(500),
            uint256(1000),
            uint256(1500),
            uint256(2000),
            uint256(2500),
            uint256(3000),
            uint256(3500),
            uint256(3775), // expected max from binary search
            uint256(4000),
            uint256(4096),
            uint256(5000),
            uint256(6000),
            uint256(7000),
            uint256(8000),
            uint256(8191),
            uint256(8192)
        ];

        console.log("================================================================");
        console.log("  _calcShapley Gas Sweep");
        console.log("  Trainers:", NUM_TRAINERS);
        console.log("  Total coalitions:", TOTAL_COALITIONS);
        console.log("  Block gas limit: 60,000,000");
        console.log("================================================================");

        for (uint256 idx = 0; idx < sampleCounts.length; idx++) {
            uint256 numSamples = sampleCounts[idx];

            vm.prank(admin);
            calculator.setEvaluationsRequired(ROUND_ID, numSamples);

            uint256 gasBefore = gasleft();
            calculator.calculateContribution(ROUND_ID, trainerIndex, NUM_TRAINERS);
            uint256 gasUsed = gasBefore - gasleft();

            bool fits = gasUsed < BLOCK_GAS_LIMIT;

            console.log("---");
            console.log("numSamples:", numSamples);
            console.log("  gasUsed:", gasUsed);
            console.log("  fitsInBlock:", fits);
            if (fits) {
                console.log("  headroom:", BLOCK_GAS_LIMIT - gasUsed);
            } else {
                console.log("  overBudgetBy:", gasUsed - BLOCK_GAS_LIMIT);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Test 2: Binary search for max numSamples within 60M gas
    // ─────────────────────────────────────────────────────────────

    function test_gasProfile_findMaxSamples() public {
        uint256 trainerIndex = 0;
        uint256 low = 1;
        uint256 high = TOTAL_COALITIONS;
        uint256 lastGood = 0;
        uint256 lastGoodGas = 0;

        while (low <= high) {
            uint256 mid = (low + high) / 2;

            vm.prank(admin);
            calculator.setEvaluationsRequired(ROUND_ID, mid);

            uint256 gasBefore = gasleft();
            calculator.calculateContribution(ROUND_ID, trainerIndex, NUM_TRAINERS);
            uint256 gasUsed = gasBefore - gasleft();

            if (gasUsed < BLOCK_GAS_LIMIT) {
                lastGood = mid;
                lastGoodGas = gasUsed;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        console.log("================================================================");
        console.log("  Binary Search: Max numSamples within 60M gas");
        console.log("  Trainers:", NUM_TRAINERS);
        console.log("================================================================");
        console.log("Max numSamples:", lastGood);
        console.log("Gas used:", lastGoodGas);
        console.log("Headroom:", BLOCK_GAS_LIMIT - lastGoodGas);

        // Measure per-iteration cost
        vm.prank(admin);
        calculator.setEvaluationsRequired(ROUND_ID, 1);
        uint256 g1Before = gasleft();
        calculator.calculateContribution(ROUND_ID, trainerIndex, NUM_TRAINERS);
        uint256 gas1 = g1Before - gasleft();

        vm.prank(admin);
        calculator.setEvaluationsRequired(ROUND_ID, lastGood);
        uint256 gNBefore = gasleft();
        calculator.calculateContribution(ROUND_ID, trainerIndex, NUM_TRAINERS);
        uint256 gasN = gNBefore - gasleft();

        uint256 perIteration = (gasN - gas1) / (lastGood - 1);
        console.log("Per-iteration gas cost:", perIteration);
        console.log("Base gas (1 sample):", gas1);

        // Also test the FIRST value that fails
        vm.prank(admin);
        calculator.setEvaluationsRequired(ROUND_ID, lastGood + 1);
        uint256 gFailBefore = gasleft();
        calculator.calculateContribution(ROUND_ID, trainerIndex, NUM_TRAINERS);
        uint256 gasFail = gFailBefore - gasleft();
        console.log("Gas at numSamples =", lastGood + 1);
        console.log("  gasUsed:", gasFail);
        console.log("  fitsInBlock:", gasFail < BLOCK_GAS_LIMIT);
    }

    // ─────────────────────────────────────────────────────────────
    //  Test 3: Full exact calculateContribution per trainer
    // ─────────────────────────────────────────────────────────────

    function test_gasProfile_allTrainers_fullExact() public {
        // Set numSamples to full 8192 (exact Shapley over all coalitions)
        vm.prank(admin);
        calculator.setEvaluationsRequired(ROUND_ID, TOTAL_COALITIONS);

        console.log("================================================================");
        console.log("  calculateContribution per Trainer (exact, all coalitions)");
        console.log("  Trainers:", NUM_TRAINERS);
        console.log("  numSamples:", TOTAL_COALITIONS);
        console.log("  Block gas limit: 60,000,000");
        console.log("================================================================");

        uint256 totalGasAllTrainers = 0;

        for (uint256 trainerIdx = 0; trainerIdx < NUM_TRAINERS; trainerIdx++) {
            uint256 gasBefore = gasleft();
            int256 shapleyValue = calculator.calculateContribution(ROUND_ID, trainerIdx, NUM_TRAINERS);
            uint256 gasUsed = gasBefore - gasleft();
            totalGasAllTrainers += gasUsed;

            console.log("---");
            console.log("Trainer:", trainerIdx);
            console.log("  Shapley value:", uint256(shapleyValue));
            console.log("  gasUsed:", gasUsed);
            console.log("  fitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);
            if (gasUsed < BLOCK_GAS_LIMIT) {
                console.log("  headroom:", BLOCK_GAS_LIMIT - gasUsed);
            } else {
                console.log("  overBudgetBy:", gasUsed - BLOCK_GAS_LIMIT);
            }
        }

        console.log("---");
        console.log("Total gas for ALL trainers:", totalGasAllTrainers);
    }

    // ─────────────────────────────────────────────────────────────
    //  Test 4: Per-sample cost extrapolation to other trainer counts
    // ─────────────────────────────────────────────────────────────

    function test_gasProfile_extrapolate() public {
        // Measure at 100 and 200 samples to get linear cost model
        vm.prank(admin);
        calculator.setEvaluationsRequired(ROUND_ID, 100);
        uint256 g100Before = gasleft();
        calculator.calculateContribution(ROUND_ID, 0, NUM_TRAINERS);
        uint256 gas100 = g100Before - gasleft();

        vm.prank(admin);
        calculator.setEvaluationsRequired(ROUND_ID, 200);
        uint256 g200Before = gasleft();
        calculator.calculateContribution(ROUND_ID, 0, NUM_TRAINERS);
        uint256 gas200 = g200Before - gasleft();

        uint256 perSample = (gas200 - gas100) / 100;
        uint256 baseGas = gas100 - (perSample * 100);

        console.log("================================================================");
        console.log("  Gas Extrapolation to Other Trainer Counts");
        console.log("  (per-sample cost measured at N=13)");
        console.log("================================================================");
        console.log("Per-sample gas cost:", perSample);
        console.log("Base overhead gas:", baseGas);
        console.log("");

        uint256[8] memory trainerCounts =
            [uint256(8), uint256(10), uint256(12), uint256(13), uint256(14), uint256(16), uint256(18), uint256(20)];

        for (uint256 i = 0; i < trainerCounts.length; i++) {
            uint256 n = trainerCounts[i];
            uint256 totalCoalitions = 1 << n;
            uint256 projectedGas = baseGas + (perSample * totalCoalitions);
            bool fits = projectedGas < BLOCK_GAS_LIMIT;

            console.log("---");
            console.log("N =", n);
            console.log("  Coalitions (2^N):", totalCoalitions);
            console.log("  Projected gas:", projectedGas);
            console.log("  Fits in 60M:", fits);
            if (fits) {
                console.log("  Headroom:", BLOCK_GAS_LIMIT - projectedGas);
            } else {
                console.log("  Over budget by:", projectedGas - BLOCK_GAS_LIMIT);
            }
            // Also compute max samples for this N within budget
            uint256 maxSamples = (BLOCK_GAS_LIMIT - baseGas) / perSample;
            if (maxSamples > totalCoalitions) maxSamples = totalCoalitions;
            console.log("  Max feasible samples:", maxSamples);
            console.log("  Feasible % of exact:", (maxSamples * 100) / totalCoalitions);
        }

        uint256 globalMaxSamples = (BLOCK_GAS_LIMIT - baseGas) / perSample;
        console.log("");
        console.log("Global max samples in 60M gas:", globalMaxSamples);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Part 1b: Actual exact Shapley gas for each N (no extrapolation)
// ═══════════════════════════════════════════════════════════════════

contract ShapleyExactCutoffTest is Test {
    uint256 constant BLOCK_GAS_LIMIT = 60_000_000;
    mapping(uint256 => bool) private _registered;

    function _deployCalculator(uint8 n) internal returns (ContributionCalculator calc, address admin) {
        admin = makeAddr("admin");
        uint256 totalCoalitions = 1 << n;
        ContributionCalculator impl = new ContributionCalculator();
        bytes memory initData =
            abi.encodeWithSelector(ContributionCalculator.initialize.selector, admin, totalCoalitions);
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        calc = ContributionCalculator(address(proxy));
    }

    function _registerRoundForBudget(
        ContributionCalculator calc,
        address admin,
        uint256 roundId,
        uint8 n,
        uint256 requestedSamples
    ) internal {
        uint256 totalCoalitions = 1 << n;
        vm.prank(admin);
        calc.setEvaluationsRequired(roundId, requestedSamples);

        uint256 emittedSamples = calc.getEvaluationsRequired(roundId, n);
        bytes32 modelHash = keccak256("cutoff_test");

        vm.startPrank(admin);
        for (uint256 sid = 0; sid < emittedSamples; sid++) {
            uint256 tgt = calc.getMask(roundId, sid, n);
            if (!_registered[tgt]) {
                calc.registerResult(roundId, sid, tgt, modelHash, int256(tgt * 100), n);
                _registered[tgt] = true;
            }
            for (uint8 bit = 0; bit < n; bit++) {
                uint256 nb = tgt ^ (1 << bit);
                if (!_registered[nb]) {
                    calc.registerResult(roundId, sid, nb, modelHash, int256(nb * 100), n);
                    _registered[nb] = true;
                }
            }
        }

        vm.stopPrank();
        for (uint256 m = 0; m < totalCoalitions; m++) {
            _registered[m] = false;
        }
    }

    function _deployAndRegister(uint8 n, uint256 roundId, uint256 requestedSamples)
        internal
        returns (ContributionCalculator)
    {
        address admin = makeAddr("admin");
        uint256 totalCoalitions = 1 << n;
        ContributionCalculator impl = new ContributionCalculator();
        bytes memory initData =
            abi.encodeWithSelector(ContributionCalculator.initialize.selector, admin, totalCoalitions);
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        ContributionCalculator calc = ContributionCalculator(address(proxy));

        _registerRoundForBudget(calc, admin, roundId, n, requestedSamples);
        return calc;
    }

    function _measureExactGas(uint8 n) internal returns (uint256 total, uint256 gasUsed) {
        total = 1 << n;
        uint256 roundId = 1;
        ContributionCalculator calc = _deployAndRegister(n, roundId, total);

        uint256 gasBefore = gasleft();
        calc.calculateContribution(roundId, 0, n);
        gasUsed = gasBefore - gasleft();
    }

    function _measureMaxSamples(uint8 n) internal returns (uint256 total, uint256 best, uint256 bestGas) {
        total = 1 << n;
        (ContributionCalculator calc, address admin) = _deployCalculator(n);

        uint256 lo = 1;
        uint256 hi = total;
        uint256 probeRoundId = 1;

        while (lo <= hi) {
            uint256 mid = (lo + hi) / 2;
            _registerRoundForBudget(calc, admin, probeRoundId, n, mid);

            uint256 gasBefore = gasleft();
            calc.calculateContribution(probeRoundId, 0, n);
            uint256 gasUsed = gasBefore - gasleft();

            if (gasUsed < BLOCK_GAS_LIMIT) {
                best = mid;
                bestGas = gasUsed;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
            probeRoundId += 1;
        }
    }

    function _emittedSamples(uint256 requestedSamples) internal pure returns (uint256) {
        return requestedSamples & ~uint256(1);
    }

    function test_exactCutoff_8to14() public {
        console.log("================================================================");
        console.log("  ACTUAL Exact Shapley Gas per N (N=8..14)");
        console.log("  Block gas limit: 60,000,000");
        console.log("================================================================");
        for (uint8 n = 8; n <= 14; n++) {
            uint256 total = 1 << n;
            uint256 roundId = 1;
            ContributionCalculator calc = _deployAndRegister(n, roundId, total);
            uint256 g0 = gasleft();
            calc.calculateContribution(roundId, 0, n);
            uint256 used = g0 - gasleft();
            bool fits = used < BLOCK_GAS_LIMIT;
            console.log("---");
            console.log("N =", uint256(n));
            console.log("  Coalitions:", total);
            console.log("  gasUsed:", used);
            console.log("  perSample:", used / total);
            console.log("  fitsIn60M:", fits);
            if (fits) {
                console.log("  headroom:", BLOCK_GAS_LIMIT - used);
                console.log("  budgetUsed%:", (used * 100) / BLOCK_GAS_LIMIT);
            } else {
                console.log("  overBudgetBy:", used - BLOCK_GAS_LIMIT);
            }
        }
    }

    function test_maxSamples_perN() public {
        console.log("================================================================");
        console.log("  Max Samples per N (N=8..14) via binary search");
        console.log("================================================================");
        for (uint8 n = 8; n <= 14; n++) {
            (uint256 total, uint256 best, uint256 bestGas) = _measureMaxSamples(n);
            uint256 emitted = _emittedSamples(best);
            console.log("---");
            console.log("N =", uint256(n));
            console.log("  totalCoalitions:", total);
            console.log("  maxRequestedSamples:", best);
            console.log("  maxEmittedSamples:", emitted);
            console.log("  gasAtMax:", bestGas);
            console.log("  exactFits:", emitted >= total);
            console.log("  coverage%:", (emitted * 100) / total);
        }
    }

    function test_maxSamples_n10() public {
        uint8 n = 10;
        (uint256 total, uint256 best, uint256 bestGas) = _measureMaxSamples(n);

        console.log("================================================================");
        console.log("  Binary Search: Max numSamples within 60M gas");
        console.log("  Trainers:", uint256(n));
        console.log("================================================================");
        console.log("Max requested samples:", best);
        console.log("Max emitted samples:", _emittedSamples(best));
        console.log("Gas used:", bestGas);
        console.log("Headroom:", BLOCK_GAS_LIMIT - bestGas);
        console.log("Coverage%:", (_emittedSamples(best) * 100) / total);
    }

    function test_exactCutoff_n10() public {
        (uint256 total, uint256 gasUsed) = _measureExactGas(10);
        console.log("N =", uint256(10));
        console.log("  Coalitions:", total);
        console.log("  gasUsed:", gasUsed);
        console.log("  perSample:", gasUsed / total);
        console.log("  fitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);
    }

    function test_exactCutoff_n11() public {
        (uint256 total, uint256 gasUsed) = _measureExactGas(11);
        console.log("N =", uint256(11));
        console.log("  Coalitions:", total);
        console.log("  gasUsed:", gasUsed);
        console.log("  perSample:", gasUsed / total);
        console.log("  fitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);
    }

    function test_exactCutoff_n12() public {
        (uint256 total, uint256 gasUsed) = _measureExactGas(12);
        console.log("N =", uint256(12));
        console.log("  Coalitions:", total);
        console.log("  gasUsed:", gasUsed);
        console.log("  perSample:", gasUsed / total);
        console.log("  fitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);
    }

    function test_exactCutoff_n13() public {
        (uint256 total, uint256 gasUsed) = _measureExactGas(13);
        console.log("N =", uint256(13));
        console.log("  Coalitions:", total);
        console.log("  gasUsed:", gasUsed);
        console.log("  perSample:", gasUsed / total);
        console.log("  fitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);
    }

    function test_exactCutoff_n14() public {
        (uint256 total, uint256 gasUsed) = _measureExactGas(14);
        console.log("N =", uint256(14));
        console.log("  Coalitions:", total);
        console.log("  gasUsed:", gasUsed);
        console.log("  perSample:", gasUsed / total);
        console.log("  fitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);
    }

    function test_exactCutoff_n15() public {
        (uint256 total, uint256 gasUsed) = _measureExactGas(15);
        console.log("N =", uint256(15));
        console.log("  Coalitions:", total);
        console.log("  gasUsed:", gasUsed);
        console.log("  perSample:", gasUsed / total);
        console.log("  fitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);
    }

    function test_exactCutoff_n16() public {
        (uint256 total, uint256 gasUsed) = _measureExactGas(16);
        console.log("N =", uint256(16));
        console.log("  Coalitions:", total);
        console.log("  gasUsed:", gasUsed);
        console.log("  perSample:", gasUsed / total);
        console.log("  fitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);
    }

    function test_maxSamples_n12() public {
        (uint256 total, uint256 best, uint256 bestGas) = _measureMaxSamples(12);
        console.log("N =", uint256(12));
        console.log("  totalCoalitions:", total);
        console.log("  maxRequestedSamples:", best);
        console.log("  maxEmittedSamples:", _emittedSamples(best));
        console.log("  gasAtMax:", bestGas);
        console.log("  coverage%:", (_emittedSamples(best) * 100) / total);
    }

    function test_maxSamples_n13() public {
        (uint256 total, uint256 best, uint256 bestGas) = _measureMaxSamples(13);
        console.log("N =", uint256(13));
        console.log("  totalCoalitions:", total);
        console.log("  maxRequestedSamples:", best);
        console.log("  maxEmittedSamples:", _emittedSamples(best));
        console.log("  gasAtMax:", bestGas);
        console.log("  coverage%:", (_emittedSamples(best) * 100) / total);
    }

    function test_maxSamples_n14() public {
        (uint256 total, uint256 best, uint256 bestGas) = _measureMaxSamples(14);
        console.log("N =", uint256(14));
        console.log("  totalCoalitions:", total);
        console.log("  maxRequestedSamples:", best);
        console.log("  maxEmittedSamples:", _emittedSamples(best));
        console.log("  gasAtMax:", bestGas);
        console.log("  coverage%:", (_emittedSamples(best) * 100) / total);
    }

    function test_maxSamples_n15() public {
        (uint256 total, uint256 best, uint256 bestGas) = _measureMaxSamples(15);
        console.log("N =", uint256(15));
        console.log("  totalCoalitions:", total);
        console.log("  maxRequestedSamples:", best);
        console.log("  maxEmittedSamples:", _emittedSamples(best));
        console.log("  gasAtMax:", bestGas);
        console.log("  coverage%:", (_emittedSamples(best) * 100) / total);
    }

    function test_maxSamples_n16() public {
        (uint256 total, uint256 best, uint256 bestGas) = _measureMaxSamples(16);
        console.log("N =", uint256(16));
        console.log("  totalCoalitions:", total);
        console.log("  maxRequestedSamples:", best);
        console.log("  maxEmittedSamples:", _emittedSamples(best));
        console.log("  gasAtMax:", bestGas);
        console.log("  coverage%:", (_emittedSamples(best) * 100) / total);
    }
}

contract DebugContributionCalculator is ContributionCalculator {
    uint256 internal constant LOG_EVERY = 1;

    function _calcShapley(uint256 roundId, uint256 trainerIndex, uint8 numberOfPlayers)
        internal
        view
        override
        returns (int256)
    {
        if (numberOfPlayers == 0) {
            return 0;
        }

        int256[] memory stratumSums = new int256[](numberOfPlayers);
        uint256[] memory stratumCounts = new uint256[](numberOfPlayers);
        uint256 numSamples = _getEvaluationsRequired(roundId, numberOfPlayers);

        console.log("debug numSamples", numSamples);
        for (uint256 i = 0; i < numSamples; ++i) {
            if (i % LOG_EVERY == 0) {
                console.log("debug i", i, "gasleft", gasleft());
            }

            uint256 generatedMask = _getMask(roundId, i, numberOfPlayers);
            uint256 playerMask = 1 << trainerIndex;

            uint256 withTrainerMask;
            uint256 withoutTrainerMask;

            if ((generatedMask & playerMask) > 0) {
                withTrainerMask = generatedMask;
                withoutTrainerMask = generatedMask & ~playerMask;
            } else {
                withoutTrainerMask = generatedMask;
                withTrainerMask = generatedMask | playerMask;
            }

            int256 withResult = getResult(roundId, withTrainerMask);
            int256 withoutResult = getResult(roundId, withoutTrainerMask);
            int256 contribution = withResult - withoutResult;
            uint256 coalitionSize = popcount(withoutTrainerMask);

            stratumSums[coalitionSize] += contribution;
            stratumCounts[coalitionSize] += 1;
        }

        int256 totalStratumAverage = 0;
        for (uint256 coalitionSize = 0; coalitionSize < numberOfPlayers; ++coalitionSize) {
            if (stratumCounts[coalitionSize] == 0) {
                continue;
            }
            totalStratumAverage += stratumSums[coalitionSize] / int256(stratumCounts[coalitionSize]);
        }

        console.log("debug completed all samples");
        return totalStratumAverage / int256(uint256(numberOfPlayers));
    }
}

contract ShapleyProgressProbeTest is Test {
    mapping(uint256 => bool) private _registered;

    function _registerRoundForBudget(
        ContributionCalculator calc,
        address admin,
        uint256 roundId,
        uint8 n,
        uint256 requestedSamples
    ) internal {
        uint256 totalCoalitions = 1 << n;
        vm.prank(admin);
        calc.setEvaluationsRequired(roundId, requestedSamples);

        uint256 emittedSamples = calc.getEvaluationsRequired(roundId, n);
        bytes32 modelHash = keccak256("progress_probe");

        vm.startPrank(admin);
        for (uint256 sid = 0; sid < emittedSamples; sid++) {
            uint256 tgt = calc.getMask(roundId, sid, n);
            if (!_registered[tgt]) {
                calc.registerResult(roundId, sid, tgt, modelHash, int256(tgt * 100), n);
                _registered[tgt] = true;
            }
            for (uint8 bit = 0; bit < n; bit++) {
                uint256 nb = tgt ^ (1 << bit);
                if (!_registered[nb]) {
                    calc.registerResult(roundId, sid, nb, modelHash, int256(nb * 100), n);
                    _registered[nb] = true;
                }
            }
        }

        vm.stopPrank();
        for (uint256 m = 0; m < totalCoalitions; m++) {
            _registered[m] = false;
        }
    }

    function _deployAndRegister(uint8 n) internal returns (ContributionCalculator calc) {
        address admin = makeAddr("admin");
        uint256 totalCoalitions = 1 << n;
        DebugContributionCalculator impl = new DebugContributionCalculator();
        bytes memory initData =
            abi.encodeWithSelector(ContributionCalculator.initialize.selector, admin, totalCoalitions);
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        calc = ContributionCalculator(address(proxy));
        _registerRoundForBudget(calc, admin, 1, n, totalCoalitions);
    }

    function test_probe_exact_n12() public {
        ContributionCalculator calc = _deployAndRegister(12);
        calc.calculateContribution(1, 0, 12);
    }

    function test_probe_exact_n13() public {
        ContributionCalculator calc = _deployAndRegister(13);
        calc.calculateContribution(1, 0, 13);
    }

    function test_probe_exact_n14() public {
        ContributionCalculator calc = _deployAndRegister(14);
        calc.calculateContribution(1, 0, 14);
    }

    function test_probe_exact_n15() public {
        ContributionCalculator calc = _deployAndRegister(15);
        calc.calculateContribution(1, 0, 15);
    }

    function test_probe_exact_n16() public {
        ContributionCalculator calc = _deployAndRegister(16);
        calc.calculateContribution(1, 0, 16);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Part 2: Full SwarmV1 end-to-end claimReward simulation
// ═══════════════════════════════════════════════════════════════════

contract ShapleyClaimRewardE2ETest is Test {
    SwarmV1 public swarm;
    ContributionCalculator public calc;

    uint8 constant NUM_TRAINERS = 13;
    uint256 constant NUM_EVALUATORS = 4;
    uint256 constant TOTAL_COALITIONS = 1 << NUM_TRAINERS; // 8192
    uint256 constant BLOCK_GAS_LIMIT = 60_000_000;
    uint256 constant NUM_SAMPLES = 837; // realistic sample budget

    address public aggregator;
    address[] public trainers;
    address[] public evaluators;

    mapping(uint256 => bool) private _registered;

    function setUp() public {
        aggregator = makeAddr("aggregator");

        // Create trainer addresses
        trainers = new address[](NUM_TRAINERS);
        for (uint256 i = 0; i < NUM_TRAINERS; i++) {
            trainers[i] = makeAddr(string(abi.encodePacked("trainer", vm.toString(i))));
        }

        // Create evaluator addresses
        evaluators = new address[](NUM_EVALUATORS);
        for (uint256 i = 0; i < NUM_EVALUATORS; i++) {
            evaluators[i] = makeAddr(string(abi.encodePacked("evaluator", vm.toString(i))));
        }

        // Deploy full SwarmV1 stack
        _deploySwarm();

        // Run the full round lifecycle: training → evaluator registration → evaluations
        _runFullRound();
    }

    function _deploySwarm() internal {
        SwarmV1 implementation = new SwarmV1();
        SelectorFactory selectorFactory = new SelectorFactory(address(this));
        CalculatorFactory calculatorFactory = new CalculatorFactory(address(this));
        AccessControlFactory accessControlFactory = new AccessControlFactory(address(this));
        CompensationFactory compensationFactory = new CompensationFactory(address(this));

        AlwaysSampled trainerSelectorImpl = new AlwaysSampled();
        RandomSampling evaluatorSelectorImpl = new RandomSampling();
        ContributionCalculator calculatorImpl = new ContributionCalculator();
        BaseAccessControl accessControlImpl = new BaseAccessControl();
        SimpleMintCompensation compensationImpl = new SimpleMintCompensation();

        (,, string memory v1,,,,) = trainerSelectorImpl.eip712Domain();
        selectorFactory.registerSelectorImplementation(address(trainerSelectorImpl));
        (,, string memory v2,,,,) = evaluatorSelectorImpl.eip712Domain();
        selectorFactory.registerSelectorImplementation(address(evaluatorSelectorImpl));
        (,, string memory v3,,,,) = calculatorImpl.eip712Domain();
        calculatorFactory.registerCalculatorImplementation(address(calculatorImpl));
        (,, string memory v4,,,,) = accessControlImpl.eip712Domain();
        accessControlFactory.registerAccessControlImplementation(address(accessControlImpl));
        (,, string memory v5,,,,) = compensationImpl.eip712Domain();
        compensationFactory.registerCompensationImplementation(address(compensationImpl));

        SwarmV1Factory factory = new SwarmV1Factory(
            address(implementation),
            address(selectorFactory),
            address(calculatorFactory),
            address(accessControlFactory),
            address(compensationFactory)
        );

        address swarmAddress = factory.getSwarmAddress(keccak256("gas-budget-salt"));

        SwarmV1Factory.SwarmParams memory params = SwarmV1Factory.SwarmParams({
            swarm: SwarmV1Factory.SwarmV1Params({name: "GasBudgetSwarm"}),
            trainerSelector: SwarmV1Factory.SelectorParams({
                id: selectorFactory.getID(v1),
                initData: abi.encodeWithSelector(AlwaysSampled.initialize.selector)
            }),
            evaluatorSelector: SwarmV1Factory.SelectorParams({
                id: selectorFactory.getID(v2),
                initData: abi.encodeWithSelector(RandomSampling.initialize.selector, 1 ether)
            }),
            contributionCalculator: SwarmV1Factory.CalculatorParams({
                id: calculatorFactory.getID(v3),
                initData: abi.encodeWithSelector(ContributionCalculator.initialize.selector, swarmAddress, NUM_SAMPLES)
            }),
            accessControl: SwarmV1Factory.AccessControlParams({
                id: accessControlFactory.getID(v4),
                initData: abi.encodeWithSelector(BaseAccessControl.initialize.selector, aggregator, trainers, evaluators)
            }),
            compensation: SwarmV1Factory.CompensationParams({
                id: compensationFactory.getID(v5),
                initData: abi.encodeWithSelector(
                    SimpleMintCompensation.initialize.selector, "GasBudgetToken", "GBT", 100000 ether, aggregator, swarmAddress
                )
            }),
            trainingPhaseConfiguration: BaseTrainingPhases.TrainingPhaseConfiguration({ttl: 1000}),
            evaluationPhaseConfiguration: BaseTrainingPhases.EvaluationPhaseConfiguration({
                ttl: 100000,
                registrationTtl: 1000
            })
        });

        swarm = SwarmV1(factory.createSwarm(keccak256("gas-budget-salt"), params));
        calc = ContributionCalculator(swarm.getContributionCalculator());
    }

    function _runFullRound() internal {
        uint256 roundId = 1;

        // ── Phase 1: Start training round ──
        vm.prank(aggregator);
        swarm.startTrainingRound();

        // ── Phase 2: All trainers register contributions ──
        for (uint256 i = 0; i < NUM_TRAINERS; i++) {
            vm.prank(trainers[i]);
            swarm.registerRoundContribution(roundId, keccak256(abi.encodePacked("model", i)));
        }

        // ── Phase 3: Fast-forward to evaluator registration ──
        BaseTrainingPhases.TrainingPhaseConfiguration memory tConfig = swarm.getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + tConfig.ttl);
        swarm.updatePhase();

        // ── Phase 4: Evaluators register ──
        for (uint256 i = 0; i < NUM_EVALUATORS; i++) {
            vm.prank(evaluators[i]);
            swarm.registerForRoundEvaluation(roundId);
        }

        // ── Phase 5: Fast-forward to evaluation phase ──
        BaseTrainingPhases.EvaluationPhaseConfiguration memory eConfig = swarm.getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + eConfig.registrationTtl);
        swarm.updatePhase();

        // ── Phase 6: Submit evaluations ──
        // Following the canonical SwarmV1.t.sol pattern:
        //   1. Each evaluator gets their assigned task via nthTaskOfNode
        //   2. Gets the target mask via calc.getMask(roundId, taskId, nTrainers)
        //   3. Calls registerEvaluation(roundId, taskId, mask, modelHash, result)
        //   4. Also registers 1-bit neighbor masks (hamming distance=1)
        //      so Shapley has both with/without trainer data
        uint8 nTrainers = uint8(swarm.getTrainerCount(roundId));
        _submitAllEvaluations(roundId, nTrainers);

        // ── Phase 7: Fast-forward to idle ──
        vm.warp(block.timestamp + eConfig.ttl);
        swarm.updatePhase();
    }

    function _submitAllEvaluations(uint256 roundId, uint8 nTrainers) internal {
        for (uint256 evalIdx = 0; evalIdx < NUM_EVALUATORS; evalIdx++) {
            vm.startPrank(evaluators[evalIdx]);

            // Get evaluator's registry ID (1-indexed)
            uint256 evalId = swarm.getEvaluatorId(roundId, evaluators[evalIdx]);

            // Iterate assigned tasks (same pattern as SwarmV1.t.sol)
            for (uint256 taskOffset = 0; taskOffset < NUM_SAMPLES; taskOffset++) {
                try swarm.nthTaskOfNode(roundId, evalId - 1, taskOffset) returns (uint256 taskId) {
                    // Get the deterministic target mask for this task
                    uint256 targetMask = calc.getMask(roundId, taskId, nTrainers);

                    // Register the target mask (distance=0), same as SwarmV1.t.sol
                    swarm.registerEvaluation(
                        roundId,
                        taskId,
                        targetMask,
                        keccak256(abi.encodePacked("model", taskId)),
                        int256(targetMask * 100)
                    );

                    // Register 1-bit neighbor masks (distance=1) so Shapley
                    // has with/without data for each trainer
                    for (uint8 bit = 0; bit < nTrainers; bit++) {
                        uint256 neighborMask = targetMask ^ (1 << bit);
                        if (!_registered[neighborMask]) {
                            swarm.registerEvaluation(
                                roundId,
                                taskId,
                                neighborMask,
                                keccak256(abi.encodePacked("model", taskId)),
                                int256(neighborMask * 100)
                            );
                            _registered[neighborMask] = true;
                        }
                    }
                    _registered[targetMask] = true;
                } catch {
                    break; // no more tasks for this evaluator
                }
            }

            vm.stopPrank();
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Test: Full claimReward through SwarmV1 for all trainers
    // ─────────────────────────────────────────────────────────────

    function test_claimReward_gasProfile_allTrainers() public {
        uint256 roundId = 1;

        console.log("================================================================");
        console.log("  SwarmV1.claimReward() Gas Profile (end-to-end)");
        console.log("  Trainers:", NUM_TRAINERS);
        console.log("  numSamples:", NUM_SAMPLES);
        console.log("  Block gas limit: 60,000,000");
        console.log("================================================================");

        uint256 totalGas = 0;

        for (uint256 i = 0; i < NUM_TRAINERS; i++) {
            uint256 gasBefore = gasleft();
            swarm.claimReward(roundId, trainers[i]);
            uint256 gasUsed = gasBefore - gasleft();
            totalGas += gasUsed;

            bool fits = gasUsed < BLOCK_GAS_LIMIT;

            console.log("---");
            console.log("Trainer:", i);
            console.log("  gasUsed:", gasUsed);
            console.log("  fitsIn60M:", fits);
            if (fits) {
                console.log("  headroom:", BLOCK_GAS_LIMIT - gasUsed);
            } else {
                console.log("  overBudgetBy:", gasUsed - BLOCK_GAS_LIMIT);
            }
        }

        console.log("---");
        console.log("Total gas for all claimReward:", totalGas);
        console.log("Average gas per claimReward:", totalGas / NUM_TRAINERS);
    }

    // ─────────────────────────────────────────────────────────────
    //  Test: calculateContribution only (for comparison)
    // ─────────────────────────────────────────────────────────────

    function test_calculateContribution_gasProfile() public {
        uint256 roundId = 1;

        console.log("================================================================");
        console.log("  ContributionCalculator.calculateContribution() (isolated)");
        console.log("  Trainers:", NUM_TRAINERS);
        console.log("  numSamples:", NUM_SAMPLES);
        console.log("================================================================");

        uint8 nTrainers = uint8(swarm.getTrainerCount(roundId));
        for (uint256 i = 0; i < NUM_TRAINERS; i++) {
            uint256 trainerRegistryId = swarm.getTrainerId(roundId, trainers[i]);

            uint256 gasBefore = gasleft();
            // Use getTrainerCount (same as claimReward does internally)
            calc.calculateContribution(roundId, trainerRegistryId - 1, nTrainers);
            uint256 gasUsed = gasBefore - gasleft();

            console.log("---");
            console.log("Trainer:", i);
            console.log("  gasUsed:", gasUsed);
            console.log("  fitsIn60M:", gasUsed < BLOCK_GAS_LIMIT);
        }
    }
}
