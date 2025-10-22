// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "forge-std/console.sol";
import {Strings} from "@openzeppelin/contracts/utils/Strings.sol";

import {SwarmV1} from "@rizemind-contracts/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "@rizemind-contracts/swarm/SwarmV1Factory.sol";
import {SelectorFactory} from "@rizemind-contracts/sampling/SelectorFactory.sol";
import {CalculatorFactory} from "@rizemind-contracts/contribution/CalculatorFactory.sol";
import {AccessControlFactory} from "@rizemind-contracts/access/AccessControlFactory.sol";
import {CompensationFactory} from "@rizemind-contracts/compensation/CompensationFactory.sol";
import {AlwaysSampled} from "@rizemind-contracts/sampling/AlwaysSampled.sol";
import {RandomSampling} from "@rizemind-contracts/sampling/RandomSampling.sol";
import {ContributionCalculator} from "@rizemind-contracts/contribution/ContributionCalculator.sol";
import {BaseAccessControl} from "@rizemind-contracts/access/BaseAccessControl.sol";
import {SimpleMintCompensation} from "@rizemind-contracts/compensation/SimpleMintCompensation.sol";
import {BaseTrainingPhases} from "@rizemind-contracts/training/BaseTrainingPhases.sol";
import {DemoParams} from "./DemoParams.sol";

contract testing is Script {
    using Strings for uint256;
    using Strings for uint8;

    SwarmV1 public testSwarm;
    BaseAccessControl public accessControl;
    SimpleMintCompensation public compensation;
    ContributionCalculator public contributionCalculator;

    address[] public trainers;
    address[] public evaluators;
    address public aggregator;

    // Predefined evaluation results (small, fixed vector)
    uint256[] public evaluationResults = [94, 71, 47, 12, 89, 65, 38, 23];
    uint256[] public trainerScores = new uint256[](DemoParams.MAX_TRAINERS);

    struct TestResult {
        uint256 numTrainers;
        uint256 evaluationsRequired;
        uint256 totalGasUsed;
        uint256 avgGasPerTrainer;
        uint256 deploymentGas;
        uint256 roundCompletionTime;
        uint256 shapleyCalculationGas;
        uint256 totalTokensDistributed;
        bool success;
        string errorMessage;
    }
    TestResult[] public testResults;

    // Function to get trainer scores
    function getTrainerScore(
        uint256 trainerIndex
    ) public view returns (uint256) {
        require(
            trainerIndex < trainerScores.length,
            "Trainer index out of bounds"
        );
        return trainerScores[trainerIndex];
    }

    // Function to get all trainer scores
    function getAllTrainerScores() public view returns (uint256[] memory) {
        return trainerScores;
    }

    function run() external {
        console.log("=== Federated Learning Scalability Test (FIXED) ===");
        console.log("Trainers %s..", DemoParams.MIN_TRAINERS.toString());
        console.log(
            "%s step %s",
            DemoParams.MAX_TRAINERS.toString(),
            DemoParams.TRAINER_STEP_SIZE.toString()
        );

        _runScalabilityTests();
        _printTestResults();
        _outputCSVData();

        console.log("=== Done ===");
    }

    // ---------- main loop ----------
    function _runScalabilityTests() internal {
        for (
            uint256 n = DemoParams.MIN_TRAINERS;
            n <= DemoParams.MAX_TRAINERS;
            n += DemoParams.TRAINER_STEP_SIZE
        ) {
            console.log("\n=== TEST %s TRAINERS ===", n.toString());
            uint256 evals = 2 ** n;
            _setupActors(n);

            uint256 g0 = gasleft();
            bool ok = _deployContracts(evals);
            uint256 deploymentGas = g0 - gasleft();

            TestResult memory r;
            if (ok) {
                r = _runTrainingDemoWithMetrics(n, evals, deploymentGas);
            } else {
                r = TestResult({
                    numTrainers: n,
                    evaluationsRequired: evals,
                    totalGasUsed: 0,
                    avgGasPerTrainer: 0,
                    deploymentGas: deploymentGas,
                    roundCompletionTime: 0,
                    shapleyCalculationGas: 0,
                    totalTokensDistributed: 0,
                    success: false,
                    errorMessage: "Deployment failed"
                });
            }

            testResults.push(r);
            console.log("Result:");
            console.log("    ok = %s", r.success ? "Y" : "N");
            console.log("    gas = %s", r.totalGasUsed.toString());
            console.log("    tokens = %s", r.totalTokensDistributed.toString());
            if (!r.success) console.log("Error: %s", r.errorMessage);

            // Display trainer scores if test was successful
            if (r.success) {
                console.log("=== TRAINER TOKEN DISTRIBUTION ===");
                for (uint256 j = 0; j < n; j++) {
                    console.log(
                        "Trainer %s: %s tokens",
                        j.toString(),
                        trainerScores[j].toString()
                    );
                }
                console.log("=== END TRAINER DISTRIBUTION ===");
            }
        }
    }

    // ---------- env setup ----------
    function _setupActors(uint256 count) internal {
        delete trainers;
        delete evaluators;

        aggregator = vm.addr(DemoParams.AGGREGATOR_KEY);
        for (uint256 i = 0; i < count; i++)
            trainers.push(vm.addr(DemoParams.TRAINER_START_KEY + i));
        for (uint256 j = 0; j < DemoParams.NUM_EVALUATORS; j++)
            evaluators.push(vm.addr(DemoParams.EVALUATOR_START_KEY + j));
    }

    // ---------- deployment ----------
    function _deployContracts(
        uint256 evaluationsRequired
    ) internal returns (bool) {
        try this._deployContractsInternal(evaluationsRequired) {
            return true;
        } catch Error(string memory reason) {
            console.log("Deploy fail: %s", reason);
            return false;
        } catch {
            console.log("Deploy fail: unknown");
            return false;
        }
    }

    function _deployContractsInternal(uint256 evaluationsRequired) external {
        SelectorFactory sf = new SelectorFactory(address(this));
        CalculatorFactory cf = new CalculatorFactory(address(this));
        AccessControlFactory af = new AccessControlFactory(address(this));
        CompensationFactory pf = new CompensationFactory(address(this));

        // impls
        AlwaysSampled alwaysS = new AlwaysSampled();
        RandomSampling randomS = new RandomSampling();
        ContributionCalculator ccImpl = new ContributionCalculator();
        BaseAccessControl acImpl = new BaseAccessControl();
        SimpleMintCompensation compImpl = new SimpleMintCompensation();
        SwarmV1 swarmImpl = new SwarmV1();

        // register
        sf.registerSelectorImplementation(address(alwaysS));
        sf.registerSelectorImplementation(address(randomS));
        cf.registerCalculatorImplementation(address(ccImpl));
        af.registerAccessControlImplementation(address(acImpl));
        pf.registerCompensationImplementation(address(compImpl));

        SwarmV1Factory factory = new SwarmV1Factory(
            address(swarmImpl),
            address(sf),
            address(cf),
            address(af),
            address(pf)
        );

        SwarmV1Factory.SwarmParams memory params = SwarmV1Factory.SwarmParams({
            swarm: SwarmV1Factory.SwarmV1Params({
                name: "ScalabilityTestSwarmFixed"
            }),
            trainerSelector: SwarmV1Factory.SelectorParams({
                id: sf.getID("always-sampled-v1.0.0"),
                initData: abi.encodeWithSelector(
                    AlwaysSampled.initialize.selector
                )
            }),
            evaluatorSelector: SwarmV1Factory.SelectorParams({
                id: sf.getID("random-sampling-v1.0.0"),
                initData: abi.encodeWithSelector(
                    RandomSampling.initialize.selector,
                    1 ether
                )
            }),
            contributionCalculator: SwarmV1Factory.CalculatorParams({
                id: cf.getID("contribution-calculator-v1.0.0"),
                initData: abi.encodeWithSelector(
                    ContributionCalculator.initialize.selector,
                    address(this),
                    evaluationsRequired < 2 ? 2 : evaluationsRequired
                )
            }),
            accessControl: SwarmV1Factory.AccessControlParams({
                id: af.getID("base-access-control-v1.0.0"),
                initData: abi.encodeWithSelector(
                    BaseAccessControl.initialize.selector,
                    aggregator,
                    trainers,
                    evaluators
                )
            }),
            compensation: SwarmV1Factory.CompensationParams({
                id: pf.getID("simple-mint-compensation-v1.0.0"),
                initData: abi.encodeWithSelector(
                    SimpleMintCompensation.initialize.selector,
                    "ScalabilityTestTokenFixed",
                    "STTF",
                    1000 ether,
                    aggregator,
                    address(this)
                )
            }),
            trainingPhaseConfiguration: BaseTrainingPhases
                .TrainingPhaseConfiguration({ttl: DemoParams.TRAINING_TTL}),
            evaluationPhaseConfiguration: BaseTrainingPhases
                .EvaluationPhaseConfiguration({
                    ttl: DemoParams.EVALUATION_TTL,
                    registrationTtl: DemoParams.EVALUATION_REGISTRATION_TTL
                })
        });

        address swarm = factory.createSwarm(
            keccak256(
                abi.encodePacked(
                    "scalability-test-swarm-fixed",
                    block.timestamp
                )
            ),
            params
        );

        testSwarm = SwarmV1(swarm);
        accessControl = BaseAccessControl(testSwarm.getAccessControl());
        compensation = SimpleMintCompensation(testSwarm.getCompensation());
        contributionCalculator = ContributionCalculator(
            testSwarm.getContributionCalculator()
        );

        // roles
        contributionCalculator.grantRole(
            contributionCalculator.DEFAULT_ADMIN_ROLE(),
            address(testSwarm)
        );
        contributionCalculator.grantRole(
            contributionCalculator.DEFAULT_ADMIN_ROLE(),
            aggregator
        );

        vm.startPrank(aggregator);
        compensation.grantRole(
            compensation.DEFAULT_ADMIN_ROLE(),
            address(this)
        );
        vm.stopPrank();

        compensation.grantRole(compensation.MINTER_ROLE(), address(testSwarm));
    }

    // ---------- one round run ----------
    function _runTrainingDemoWithMetrics(
        uint256 numTrainers,
        uint256 evaluationsRequired,
        uint256 deploymentGas
    ) internal returns (TestResult memory) {
        uint256 t0 = block.timestamp;
        uint256 totalGasUsed;
        uint256 shapleyGas;
        uint256 tokens;

        bool ok = true;
        string memory err = "";

        try this._runCompleteTrainingDemoWithGasTracking() returns (
            uint256 g,
            uint256 sg,
            uint256 td
        ) {
            totalGasUsed = g;
            shapleyGas = sg;
            tokens = td;
        } catch Error(string memory reason) {
            ok = false;
            err = reason;
        } catch {
            ok = false;
            err = "Unknown error";
        }

        uint256 roundTime = block.timestamp - t0;
        uint256 avgGas = numTrainers == 0 ? 0 : totalGasUsed / numTrainers;

        return
            TestResult({
                numTrainers: numTrainers,
                evaluationsRequired: evaluationsRequired,
                totalGasUsed: totalGasUsed,
                avgGasPerTrainer: avgGas,
                deploymentGas: deploymentGas,
                roundCompletionTime: roundTime,
                shapleyCalculationGas: shapleyGas,
                totalTokensDistributed: tokens,
                success: ok,
                errorMessage: err
            });
    }

    function _runCompleteTrainingDemoWithGasTracking()
        external
        returns (
            uint256 totalGasUsed,
            uint256 shapleyGas,
            uint256 totalTokensDistributed
        )
    {
        (uint256 g, uint256 sg, uint256 td) = _runTrainingRoundWithGasTracking(
            1,
            "Scalability Test Round"
        );
        return (g, sg, td);
    }

    // ---------- steps (gas-tracked) ----------
    function _runTrainingRoundWithGasTracking(
        uint256 roundId,
        string memory /*description*/
    )
        internal
        returns (
            uint256 totalGasUsed,
            uint256 shapleyGas,
            uint256 totalTokensDistributed
        )
    {
        uint256 g;

        // 1) start
        vm.prank(aggregator);
        g = gasleft();
        testSwarm.startTrainingRound();
        totalGasUsed += g - gasleft();

        // 2) trainers
        totalGasUsed += _registerTrainersForRoundWithGas(roundId);

        // 3) -> evaluation registration window
        totalGasUsed += _advancePhaseWithWarp(
            testSwarm.getTrainingPhaseConfiguration().ttl + 1
        );

        // 4) evaluators
        totalGasUsed += _registerEvaluatorsForRoundWithGas(roundId);

        // 5) -> evaluation
        totalGasUsed += _advancePhaseWithWarp(
            testSwarm.getEvaluationPhaseConfiguration().registrationTtl + 1
        );

        // 6) evaluations (+ shapley)
        (g, shapleyGas) = _registerEvaluationsForRoundWithGas(roundId);
        totalGasUsed += g;

        // 7) complete
        totalGasUsed += _advancePhaseWithWarp(
            testSwarm.getEvaluationPhaseConfiguration().ttl + 1
        );

        // 8) claim
        (g, totalTokensDistributed) = _claimRewardsWithGas(roundId);
        totalGasUsed += g;
    }

    function _advancePhaseWithWarp(
        uint256 secs
    ) internal returns (uint256 gasUsed) {
        vm.warp(block.timestamp + secs);
        vm.prank(aggregator);
        uint256 g = gasleft();
        testSwarm.updatePhase();
        gasUsed = g - gasleft();
    }

    function _registerTrainersForRoundWithGas(
        uint256 roundId
    ) internal returns (uint256 totalGas) {
        for (uint256 i = 0; i < trainers.length; i++) {
            vm.prank(trainers[i]);
            uint256 g = gasleft();
            testSwarm.registerRoundContribution(
                roundId,
                keccak256(abi.encodePacked("model", i + 1, roundId))
            );
            totalGas += g - gasleft();
        }
    }

    function _registerEvaluatorsForRoundWithGas(
        uint256 roundId
    ) internal returns (uint256 totalGas) {
        for (uint256 i = 0; i < evaluators.length; i++) {
            vm.prank(evaluators[i]);
            uint256 g = gasleft();
            testSwarm.registerForRoundEvaluation(roundId);
            totalGas += g - gasleft();
        }
    }

    function _registerEvaluationsForRoundWithGas(
        uint256 roundId
    ) internal returns (uint256 totalGas, uint256 shapleyGas) {
        uint256 n = testSwarm.getTrainerCount(roundId);
        uint256 required = 2 ** n;

        vm.prank(aggregator);
        uint256 g = gasleft();
        contributionCalculator.setEvaluationsRequired(roundId, required);
        g -= gasleft();
        totalGas += g;
        shapleyGas += g;

        for (uint256 i = 0; i < required; i++) {
            // Use round-robin assignment of evaluators
            uint256 evaluatorIndex = i % evaluators.length;
            uint256 evalId = testSwarm.getEvaluatorId(
                roundId,
                evaluators[evaluatorIndex]
            );
            uint256 taskId = testSwarm.nthTaskOfNode(roundId, evalId - 1, 0);
            uint256 mask = contributionCalculator.getMask(
                roundId,
                taskId,
                uint8(n)
            );

            int256 score = int256(
                i < evaluationResults.length
                    ? evaluationResults[i]
                    : uint256(
                        keccak256(
                            abi.encodePacked(
                                "score",
                                i,
                                roundId,
                                block.timestamp
                            )
                        )
                    ) % 101
            );

            console.log("Score for evaluation %s ", score);

            vm.prank(evaluators[evaluatorIndex]);
            uint256 g2 = gasleft();
            testSwarm.registerEvaluation(
                roundId,
                taskId,
                mask,
                keccak256(abi.encodePacked("evaluation", i + 1, roundId)),
                score
            );
            g2 -= gasleft();
            totalGas += g2;
            shapleyGas += g2;
        }
    }

    function _claimRewardsWithGas(
        uint256 roundId
    ) internal returns (uint256 totalGas, uint256 totalTokens) {
        uint256 n = testSwarm.getTrainerCount(roundId);
        console.log("=== TOKENS CLAIMED PER TRAINER ===");
        for (uint256 i = 0; i < n; i++) {
            uint256 beforeBal = compensation.balanceOf(trainers[i]);
            vm.prank(trainers[i]);
            uint256 g = gasleft();
            testSwarm.claimReward(uint64(roundId), trainers[i]);
            totalGas += g - gasleft();
            uint256 gained = compensation.balanceOf(trainers[i]) - beforeBal;
            trainerScores[i] = gained;
            totalTokens += gained;

            console.log(
                "Trainer %s: %s tokens claimed",
                i.toString(),
                gained.toString()
            );
        }
        console.log("Total tokens distributed: %s", totalTokens.toString());
        console.log("=== END TOKENS CLAIMED ===");
    }

    // ---------- reporting ----------
    function _printTestResults() internal view {
        console.log("\n=== SUMMARY ===");
        uint256 okCount;
        uint256 totalGas;
        uint256 maxGas;
        uint256 minGas = type(uint256).max;
        uint256 tokens;

        for (uint256 i = 0; i < testResults.length; i++) {
            TestResult memory r = testResults[i];
            if (r.success) {
                okCount++;
                totalGas += r.totalGasUsed;
                tokens += r.totalTokensDistributed;
                if (r.totalGasUsed > maxGas) maxGas = r.totalGasUsed;
                if (r.totalGasUsed < minGas) minGas = r.totalGasUsed;
            }
            // Split up the console log lines instead of all in one
            console.log("TestResult:");
            console.log("  Trainers = %s", r.numTrainers.toString());
            console.log("  ok = %s", r.success ? "Y" : "N");
            console.log("  gas = %s", r.totalGasUsed.toString());
            console.log("  avg = %s", r.avgGasPerTrainer.toString());
            console.log("  tokens = %s", r.totalTokensDistributed.toString());
        }

        // Split up the summary statistics too
        console.log("Summary:");
        console.log(
            "  Success %% = %s",
            ((okCount * 100) / testResults.length).toString()
        );
        console.log("  TotalGas = %s", totalGas.toString());
        console.log("  Tokens = %s", tokens.toString());
        console.log("  MaxGas = %s", maxGas.toString());
        console.log(
            "  MinGas = %s",
            (minGas == type(uint256).max ? 0 : minGas).toString()
        );
    }

    function _outputCSVData() internal view {
        console.log("\n=== CSV ===");
        // Keep table headers as-is since they're a header row, not multiple values
        console.log(
            "| numTrainers | evaluationsRequired | totalGasUsed | avgGasPerTrainer | deploymentGas | roundCompletionTime | shapleyCalculationGas | totalTokensDistributed | success |"
        );
        console.log(
            "|-------------|--------------------|--------------|------------------|---------------|---------------------|-----------------------|------------------------|---------|"
        );
        for (uint256 i = 0; i < testResults.length; i++) {
            TestResult memory r = testResults[i];
            // Instead of one log with all columns, do one column at a time in order
            console.log("numTrainers = %s", r.numTrainers.toString());
            console.log(
                "evaluationsRequired = %s",
                r.evaluationsRequired.toString()
            );
            console.log("totalGasUsed = %s", r.totalGasUsed.toString());
            console.log("avgGasPerTrainer = %s", r.avgGasPerTrainer.toString());
            console.log("deploymentGas = %s", r.deploymentGas.toString());
            console.log(
                "roundCompletionTime = %s",
                r.roundCompletionTime.toString()
            );
            console.log(
                "shapleyCalculationGas = %s",
                r.shapleyCalculationGas.toString()
            );
            console.log(
                "totalTokensDistributed = %s",
                r.totalTokensDistributed.toString()
            );
            console.log("success = %s", r.success ? "true" : "false");
            // You may choose to add a separator for readability (not mandatory)
            // console.log("-------------------------");
        }
    }
}
