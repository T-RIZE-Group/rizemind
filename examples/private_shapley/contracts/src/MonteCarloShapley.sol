// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title ImprovedPrivateShapley – Monte Carlo Shapley approximation
 * @notice Uses Monte Carlo sampling to approximate Shapley values with O(n*k) complexity
 * @author Blockchain Dev Team
 */
contract MonteCarloPrivateShapley is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    /*──────────────────────────────────────────────────────────────────
                                CONSTANTS
    ──────────────────────────────────────────────────────────────────*/
    uint8 public constant MAX_TRAINERS = 255;
    uint8 public constant MAX_SHAPLEY_PLAYERS = 255; // increased to 20
    uint256 public constant PRECISION = 1e6;
    uint256 public constant MAX_BATCH_SIZE = 256;
    uint256 public constant COMMIT_REVEAL_WINDOW = 7 days;
    uint256 public constant MIN_RESULT_THRESHOLD = 1;

    // Monte Carlo parameters
    uint256 public constant SAMPLES_PER_PLAYER = 150; // Tunable based on accuracy needs
    uint256 public constant SEED_UPDATE_INTERVAL = 256; // Update seed every N blocks

    /*──────────────────────────────────────────────────────────────────
                                  TOKEN
    ──────────────────────────────────────────────────────────────────*/
    IERC20 public immutable rewardToken;

    /*──────────────────────────────────────────────────────────────────
                            TRAINER & TESTER REGISTRY
    ──────────────────────────────────────────────────────────────────*/
    mapping(address => bool) public isRegisteredTrainer;
    mapping(address => bool) public isTester;

    event TrainerAllowListed(address indexed trainer, bool allowed);
    event TesterRegistered(address indexed tester, bool enabled);

    /*──────────────────────────────────────────────────────────────────
                                MODIFIERS
    ──────────────────────────────────────────────────────────────────*/
    modifier maxBatchSize(uint256 length) {
        require(length > 0, "Batch cannot be empty");
        require(length <= MAX_BATCH_SIZE, "Batch size exceeds maximum");
        _;
    }

    modifier onlyTester() {
        require(isTester[msg.sender], "caller is not a tester");
        _;
    }

    constructor(address _token) Ownable(msg.sender) {
        require(_token != address(0), "token=0");
        rewardToken = IERC20(_token);
    }

    /*──────────────────────────────────────────────────────────────────
                         TRAINER REGISTRATION (global allow-list)
    ──────────────────────────────────────────────────────────────────*/
    function setTrainers(
        address[] calldata addrs,
        bool[] calldata flags
    ) external onlyOwner maxBatchSize(addrs.length) {
        require(addrs.length == flags.length, "!len");
        for (uint256 i; i < addrs.length; ) {
            require(addrs[i] != address(0), "Invalid trainer address");
            isRegisteredTrainer[addrs[i]] = flags[i];
            emit TrainerAllowListed(addrs[i], flags[i]);
            unchecked {
                ++i;
            }
        }
    }

    /*──────────────────────────────────────────────────────────────────
                              TESTER MANAGEMENT
    ──────────────────────────────────────────────────────────────────*/
    function setTesters(
        address[] calldata testers,
        bool[] calldata flags
    ) external onlyOwner maxBatchSize(testers.length) {
        require(testers.length == flags.length, "Array lengths must match");

        for (uint256 i = 0; i < testers.length; ++i) {
            address tester = testers[i];
            bool flag = flags[i];

            require(tester != address(0), "Invalid tester address");
            isTester[tester] = flag;
            emit TesterRegistered(tester, flag);
        }
    }

    /*──────────────────────────────────────────────────────────────────
                        ROUND BASICS & LIFECYCLE
    ──────────────────────────────────────────────────────────────────*/
    struct Round {
        uint256 start;
        uint256 end;
        bool active;
    }
    mapping(uint256 => Round) public rounds;
    event RoundDefined(uint256 id, uint256 start, uint256 end, bool active);

    modifier onlyActiveRound(uint256 rid) {
        Round memory r = rounds[rid];
        require(r.active, "Round is not active");
        require(
            block.timestamp >= r.start && block.timestamp <= r.end,
            "Round is not in progress"
        );
        _;
    }

    function createRound(
        uint256 rid,
        uint256 start,
        uint256 end
    ) external onlyOwner {
        require(start < end, "End time must be after start time");
        require(!rounds[rid].active, "Round already exists");
        rounds[rid] = Round(start, end, true);
        emit RoundDefined(rid, start, end, true);
    }

    function updateRound(
        uint256 rid,
        uint256 start,
        uint256 end,
        bool active
    ) external onlyOwner {
        require(rounds[rid].active, "Round does not exist");
        require(start < end, "End time must be after start time");
        rounds[rid] = Round(start, end, active);
        emit RoundDefined(rid, start, end, active);
    }

    /*──────────────────────────────────────────────────────────────────
                    1.  DYNAMIC TRAINER-INDEX MAPPING (per round)
    ──────────────────────────────────────────────────────────────────*/
    struct MappingCommit {
        bytes32 commitment;
        bool revealed;
    }
    mapping(uint256 => MappingCommit) public mappingCommit;
    mapping(uint256 => mapping(address => uint8)) public roundAddrToIdx;
    mapping(uint256 => mapping(uint8 => address)) public roundIdxToAddr;
    mapping(uint256 => mapping(address => bytes32)) private roundAddrSalt;
    mapping(uint256 => uint8) public roundTrainerCount;
    mapping(uint256 => mapping(uint256 => bytes32)) private roundMaskToCid;

    event MappingCommitted(uint256 indexed round, bytes32 commitment);
    event MappingRevealed(uint256 indexed round, uint8 nTrainers);

    function commitTrainerMapping(
        uint256 rid,
        bytes32 commitment
    ) external onlyOwner {
        require(mappingCommit[rid].commitment == 0, "already committed");
        mappingCommit[rid].commitment = commitment;
        emit MappingCommitted(rid, commitment);
    }

    function revealTrainerMapping(
        uint256 rid,
        address[] calldata addrs,
        bytes32[] calldata salts
    ) external onlyOwner {
        require(
            addrs.length == salts.length && addrs.length <= MAX_TRAINERS,
            "!len"
        );
        MappingCommit storage mc = mappingCommit[rid];
        require(mc.commitment != 0 && !mc.revealed, "!commit");

        bytes32 calc = keccak256(abi.encodePacked(addrs, salts));
        require(calc == mc.commitment, "commit mismatch");

        for (uint8 i = 0; i < addrs.length; ++i) {
            address a = addrs[i];
            require(isRegisteredTrainer[a], "not allow-listed");
            uint8 idx = i + 1;
            roundAddrToIdx[rid][a] = idx;
            roundIdxToAddr[rid][idx] = a;
            roundAddrSalt[rid][a] = salts[i];
        }
        roundTrainerCount[rid] = uint8(addrs.length);
        mc.revealed = true;
        emit MappingRevealed(rid, uint8(addrs.length));
    }

    /*──────────────────────────────────────────────────────────────────
                         2.  COALITIONS (commit-reveal)
    ──────────────────────────────────────────────────────────────────*/
    struct Coalition {
        bytes32 commitment;
        bytes32 bitfield;
        bytes32 nonce;
        uint256 result;
        bool committed;
        bool revealed;
        uint256 revealDeadline;
        uint256 sumScores;
        uint256 numScores;
    }

    mapping(bytes32 => Coalition) public coalitionData;
    mapping(bytes32 => mapping(address => uint256)) public testerResults;
    mapping(bytes32 => address[]) public coalitionTesters;

    event CoalitionCommitted(
        bytes32 id,
        bytes32 commitment,
        uint256 revealDeadline
    );
    event CoalitionRevealed(bytes32 id, bytes32 bitfield, bytes32 nonce);
    event ResultPublished(bytes32 id, uint256 score, address tester);

    function commitCoalitions(
        uint256 roundId,
        bytes32[] calldata ids,
        bytes32[] calldata commitments
    ) external onlyOwner onlyActiveRound(roundId) maxBatchSize(ids.length) {
        require(ids.length == commitments.length, "Array lengths must match");

        for (uint256 i = 0; i < ids.length; ++i) {
            Coalition storage c = coalitionData[ids[i]];
            require(!c.committed, "Coalition already committed");

            uint256 revealDeadline = block.timestamp + COMMIT_REVEAL_WINDOW;
            c.commitment = commitments[i];
            c.committed = true;
            c.revealDeadline = revealDeadline;

            emit CoalitionCommitted(ids[i], commitments[i], revealDeadline);
        }
    }

    function revealCoalitions(
        uint256 roundId,
        bytes32[] calldata ids,
        bytes32[] calldata bitfields,
        bytes32[] calldata nonces
    ) external onlyOwner onlyActiveRound(roundId) maxBatchSize(ids.length) {
        require(
            ids.length == bitfields.length && ids.length == nonces.length,
            "Array lengths must match"
        );

        for (uint256 i = 0; i < ids.length; ++i) {
            Coalition storage c = coalitionData[ids[i]];
            require(
                c.committed,
                "Coalition must be committed before revealing"
            );
            require(!c.revealed, "Coalition has already been revealed");
            require(
                block.timestamp <= c.revealDeadline,
                "Reveal deadline has passed"
            );

            require(
                keccak256(abi.encodePacked(bitfields[i], nonces[i])) ==
                    c.commitment,
                "Invalid commitment: bitfield and nonce do not match original commitment"
            );

            require(
                uint256(bitfields[i]) > 0,
                "Coalition must include at least one trainer"
            );

            c.bitfield = bitfields[i];
            roundMaskToCid[roundId][uint256(c.bitfield)] = ids[i];
            c.nonce = nonces[i];
            c.revealed = true;
            emit CoalitionRevealed(ids[i], bitfields[i], nonces[i]);
        }
    }

    function publishResults(
        uint256 roundId,
        bytes32[] calldata ids,
        uint256[] calldata scores
    ) external onlyTester onlyActiveRound(roundId) maxBatchSize(ids.length) {
        require(ids.length == scores.length, "Array lengths must match");

        for (uint256 i = 0; i < ids.length; ++i) {
            require(
                testerResults[ids[i]][msg.sender] == 0,
                "Tester has already submitted"
            );

            bytes32 cid = ids[i];
            uint256 score = scores[i];

            require(
                score >= MIN_RESULT_THRESHOLD,
                "Score below minimum threshold"
            );

            Coalition storage c = coalitionData[cid];
            require(c.committed, "Coalition not committed");

            c.sumScores += score;
            c.numScores += 1;
            coalitionTesters[cid].push(msg.sender);

            testerResults[cid][msg.sender] = score;

            c.result = c.sumScores / c.numScores;

            emit ResultPublished(cid, score, msg.sender);
        }
    }

    /*──────────────────────────────────────────────────────────────────
              3.  MONTE CARLO SHAPLEY APPROXIMATION
    ──────────────────────────────────────────────────────────────────*/
    mapping(uint256 => mapping(uint256 => uint256)) public roundCoalitionValues;
    mapping(uint256 => mapping(uint256 => bool)) public roundCoalitionExists;
    mapping(uint256 => uint256[]) private roundAllCoalitions;

    event ShapleyCoalitionValueSet(
        uint256 roundId,
        uint256 mask,
        uint256 value
    );

    // Random number generation for Monte Carlo sampling
    mapping(uint256 => uint256) private roundRandomSeed;

    /**
     * @notice Generate a pseudo-random number using block data and internal state
     * @param roundId The round ID
     * @param iteration Current iteration number
     * @param extraSeed Additional seed data
     * @return A pseudo-random uint256
     */
    function _getRandom(
        uint256 roundId,
        uint256 iteration,
        uint256 extraSeed
    ) private view returns (uint256) {
        // Use a combination of block data and internal state for randomness
        // In production, consider using Chainlink VRF for better randomness
        return
            uint256(
                keccak256(
                    abi.encodePacked(
                        blockhash(block.number - 1),
                        roundId,
                        iteration,
                        extraSeed,
                        roundRandomSeed[roundId],
                        block.timestamp,
                        block.prevrandao // Use prevrandao for additional entropy
                    )
                )
            );
    }

    /**
     * @notice Generate a random coalition mask for Monte Carlo sampling
     * @param roundId The round ID
     * @param playerBit The bit representing the player (must be included)
     * @param nPlayers Total number of players
     * @param iteration Current sampling iteration
     * @return mask The coalition mask with the player included
     */
    function _generateRandomCoalition(
        uint256 roundId,
        uint256 playerBit,
        uint8 nPlayers,
        uint256 iteration
    ) private view returns (uint256 mask) {
        uint256 rand = _getRandom(roundId, iteration, playerBit);

        // Start with the player included
        mask = playerBit;

        // For each other player, randomly decide inclusion
        for (uint8 i = 0; i < nPlayers; i++) {
            uint256 bit = 1 << i;
            if (bit != playerBit) {
                // Use different bits of the random number for each player
                if ((rand >> i) & 1 == 1) {
                    mask |= bit;
                }
            }
        }
    }

    // /**
    //  * @notice Calculate Shapley value using Monte Carlo approximation
    //  * @param roundId The round ID
    //  * @param trainerBit The bit mask for the trainer
    //  * @param nPlayers Total number of players
    //  * @return total The approximated Shapley value
    //  */
    // function _calcShapleyValue(
    //     uint256 roundId,
    //     uint256 trainerBit,
    //     uint8 nPlayers
    // ) private view returns (int256 total) {
    //     require(nPlayers > 0, "No players");

    //     // For single player, return their solo value
    //     if (nPlayers == 1) {
    //         bytes32 cid = roundMaskToCid[roundId][trainerBit];
    //         return int256(coalitionData[cid].result);
    //     }

    //     uint256 sumMarginal = 0;
    //     uint256 validSamples = 0;

    //     // Monte Carlo sampling
    //     for (uint256 i = 0; i < SAMPLES_PER_PLAYER; i++) {
    //         // Generate random coalition containing the player
    //         uint256 coalitionWith = _generateRandomCoalition(
    //             roundId,
    //             trainerBit,
    //             nPlayers,
    //             i
    //         );

    //         // Coalition without the player
    //         uint256 coalitionWithout = coalitionWith & ~trainerBit;

    //         // Get coalition IDs
    //         bytes32 cidWith = roundMaskToCid[roundId][coalitionWith];
    //         bytes32 cidWithout = roundMaskToCid[roundId][coalitionWithout];

    //         // Skip if either coalition doesn't exist
    //         if (cidWith == bytes32(0) || cidWithout == bytes32(0)) {
    //             continue;
    //         }

    //         // Get values
    //         uint256 valueWith = coalitionData[cidWith].result;
    //         uint256 valueWithout = coalitionData[cidWithout].result;

    //         // Calculate marginal contribution
    //         if (valueWith >= valueWithout) {
    //             sumMarginal += (valueWith - valueWithout);
    //         } else {
    //             // Handle negative contributions - use safe subtraction
    //             if (sumMarginal >= (valueWithout - valueWith)) {
    //                 sumMarginal -= (valueWithout - valueWith);
    //             } else {
    //                 sumMarginal = 0; // Prevent underflow
    //             }
    //         }

    //         validSamples++;
    //     }

    //     // Average the marginal contributions
    //     if (validSamples > 0) {
    //         total = int256(sumMarginal / validSamples);
    //     }
    // }

    /**
     * @notice Calculate Shapley value using Monte Carlo approximation
     * @param roundId The round ID
     * @param trainerBit The bit mask for the trainer
     * @param nPlayers Total number of players
     * @return total The approximated Shapley value
     */
    function _calcShapleyValuePerm(
        uint256 roundId,
        uint256 trainerBit,
        uint8 nPlayers
    ) private view returns (int256 total) {
        require(nPlayers > 0, "No players");

        // For single player, return their solo value
        if (nPlayers == 1) {
            bytes32 cid = roundMaskToCid[roundId][trainerBit];
            return int256(coalitionData[cid].result);
        }

        int256 weightedSum = 0;
        uint256 totalWeight = 0;
        uint256 validSamples = 0;

        uint256 totalLoops = SAMPLES_PER_PLAYER * nPlayers;

        // Monte Carlo sampling
        for (uint256 i = 0; i < totalLoops; i++) {
            // Generate random coalition containing the player
            uint256 coalitionWith = _generateRandomCoalition(
                roundId,
                trainerBit,
                nPlayers,
                i
            );

            // Coalition without the player
            uint256 coalitionWithout = coalitionWith & ~trainerBit;

            // Get coalition IDs
            bytes32 cidWith = roundMaskToCid[roundId][coalitionWith];
            bytes32 cidWithout = roundMaskToCid[roundId][coalitionWithout];

            // Skip if either coalition doesn't exist
            if (cidWith == bytes32(0) || cidWithout == bytes32(0)) {
                continue;
            }

            // Get values
            uint256 valueWith = coalitionData[cidWith].result;
            uint256 valueWithout = coalitionData[cidWithout].result;

            // Calculate the size of the coalition S (including the trainer)
            uint8 sizeS = _popcnt(coalitionWith);

            // Calculate weight = (|S|-1)! * (n-|S|)! / n!
            uint256 weight = (_factorial[sizeS - 1] *
                _factorial[nPlayers - sizeS] *
                PRECISION) / _factorial[nPlayers];

            // Calculate weighted marginal contribution
            int256 marginal = int256(valueWith) - int256(valueWithout);
            weightedSum += (marginal * int256(weight)) / int256(PRECISION);
            totalWeight += weight;
            validSamples++;
        }

        // Return weighted average
        if (validSamples > 0 && totalWeight > 0) {
            total = (weightedSum * int256(PRECISION)) / int256(totalWeight);
        }
    }

    function _calcShapleyValue(
        uint256 roundId,
        uint256 trainerBit,
        uint8 nPlayers
    ) private view returns (int256 total) {
        // require(nPlayers <= MAX_SHAPLEY_PLAYERS, ">20 players");

        // Find which trainer index this bit represents (1-based)
        uint8 trainerIdx = 0;
        for (uint8 i = 0; i < nPlayers; i++) {
            if (trainerBit == (1 << i)) {
                trainerIdx = i + 1;
                break;
            }
        }
        require(trainerIdx > 0, "Invalid trainer bit");

        // Iterate through all possible coalitions
        uint256 maxMask = (1 << nPlayers) - 1;

        for (uint256 mask = 0; mask <= maxMask; mask++) {
            // Skip if this coalition doesn't include our trainer
            if ((mask & trainerBit) == 0) continue;

            // Calculate marginal contribution
            uint256 coalitionWith = mask;
            uint256 coalitionWithout = mask & ~trainerBit;

            bytes32 cidWith = roundMaskToCid[roundId][coalitionWith];
            bytes32 cidWithout = roundMaskToCid[roundId][coalitionWithout];

            uint256 valueWith = coalitionData[cidWith].result;
            uint256 valueWithout = coalitionData[cidWithout].result;

            // Calculate the size of the coalition S (including the trainer)
            uint8 sizeS = _popcnt(coalitionWith);

            // Weight = (|S|-1)! * (n-|S|)! / n!
            uint256 weight = (_factorial[sizeS - 1] *
                _factorial[nPlayers - sizeS] *
                PRECISION) / _factorial[nPlayers];

            // Add weighted marginal contribution
            int256 marginal = int256(valueWith) - int256(valueWithout);
            total += (marginal * int256(weight)) / int256(PRECISION);
        }
    }

    uint256[58] private _factorial = [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
        51090942171709440000,
        1124000727777607680000,
        25852016738884976640000,
        620448401733239439360000,
        15511210043330985984000000,
        403291461126605635584000000,
        10888869450418352160768000000,
        304888344611713860501504000000,
        8841761993739701954543616000000,
        265252859812191058636308480000000,
        8222838654177922817725562880000000,
        263130836933693530167218012160000000,
        8683317618811886495518194401280000000,
        295232799039604140847618609643520000000,
        10333147966386144929666651337523200000000,
        371993326789901217467999448150835200000000,
        13763753091226345046315979581580902400000000,
        523022617466601111760007224100074291200000000,
        20397882081197443358640281739902897356800000000,
        815915283247897734345611269596115894272000000000,
        33452526613163807108170062053440751665152000000000,
        1405006117752879898543142606244511569936384000000000,
        60415263063373835637355132068513997507264512000000000,
        2658271574788448768043625811014615890319638528000000000,
        119622220865480194561963161495657715064383733760000000000,
        5502622159812088949850305428800254892961651752960000000000,
        258623241511168180642964355153611979969197632389120000000000,
        12413915592536072670862289047373375038521486354677760000000000,
        608281864034267560872252163321295376887552831379210240000000000,
        30414093201713378043612608166064768844377641568960512000000000000,
        1551118753287382280224243016469303211063259720016986112000000000000,
        80658175170943878571660636856403766975289505440883277824000000000000,
        4274883284060025564298013753389399649690343788366813724672000000000000,
        230843697339241380472092742683027581083278564571807941132288000000000000,
        12696403353658275925965100847566516959580321051449436762275840000000000000,
        710998587804863451854045647463724949736497978881168166272498688000000000000,
        40526919504877216755680601905432322134980384796226602145184481280000000000000
    ];

    /**
     * @notice Update the random seed for a round (owner only)
     * @param roundId The round ID
     * @param seed New seed value
     */
    function updateRoundSeed(uint256 roundId, uint256 seed) external onlyOwner {
        roundRandomSeed[roundId] = seed;
    }

    /**
     * @notice Get the number of coalitions available for sampling
     * @param roundId The round ID
     * @return count Number of revealed coalitions
     */
    function getCoalitionCount(
        uint256 roundId
    ) external view returns (uint256 count) {
        return roundAllCoalitions[roundId].length;
    }

    /*──────────────────────────────────────────────────────────────────
                              4.  CLAIM REWARDS
    ──────────────────────────────────────────────────────────────────*/
    mapping(uint256 => mapping(address => bool)) public roundRewardClaimed;
    mapping(uint256 => mapping(address => int256)) public cachedShapleyValues;
    mapping(uint256 => mapping(address => bool)) public shapleyComputed;

    event RewardClaimed(address trainer, uint256 roundId, uint256 amount);
    event ShapleyValueComputed(address trainer, uint256 roundId, int256 value);

    function claimRewards(
        uint256 rid,
        bytes32[] calldata coalitionIds,
        bytes32 salt
    ) external maxBatchSize(coalitionIds.length) nonReentrant {
        require(rounds[rid].active, "Round is not active");
        require(coalitionIds.length > 0, "No coalitions to claim");

        uint8 idx = roundAddrToIdx[rid][msg.sender];
        require(idx != 0, "not mapped");
        require(salt == roundAddrSalt[rid][msg.sender], "bad salt");

        uint256 idxMask = 1 << (idx - 1);
        uint8 nPlayers = roundTrainerCount[rid];
        require(
            nPlayers > 0 && nPlayers <= MAX_SHAPLEY_PLAYERS,
            "Invalid player count"
        );

        uint256 maxMask = (1 << nPlayers) - 1;

        uint256 totalLoops = SAMPLES_PER_PLAYER * nPlayers;

        int256 shapley;
        if (totalLoops > maxMask) {
            shapley = _calcShapleyValuePerm(rid, idxMask, nPlayers);
        } else {
            shapley = _calcShapleyValue(rid, idxMask, nPlayers);
        }
        require(shapley >= 0, "Non-positive Shapley value");

        uint256 shapleyMul = uint256(shapley);
        uint256 tot;

        roundRewardClaimed[rid][msg.sender] = true;

        // TODO: decide how to handle the reward.
        //     uint256 reward = (c.result * shapleyMul) / PRECISION;
        //     tot += reward;
        //     emit RewardClaimed(msg.sender, cid, reward);
    }

    /*──────────────────────────────────────────────────────────────────
                              VIEW HELPERS
    ──────────────────────────────────────────────────────────────────*/
    function isTrainerInCoalition(
        bytes32 id,
        address trainer,
        uint256 roundId
    ) external view returns (bool) {
        uint8 tIndex = roundAddrToIdx[roundId][trainer];
        if (tIndex == 0) return false;

        Coalition storage c = coalitionData[id];
        if (!c.revealed) return false;

        return (uint256(c.bitfield) >> (tIndex - 1)) & 1 == 1;
    }

    function getCoalitionResult(
        bytes32 id
    ) external view returns (uint256 score, uint256 testerCount) {
        return (coalitionData[id].result, coalitionTesters[id].length);
    }

    function getTrainerShapleyValue(
        uint256 roundId,
        address trainer
    ) external view returns (int256 value, bool isPositive) {
        if (shapleyComputed[roundId][trainer]) {
            value = cachedShapleyValues[roundId][trainer];
        } else {
            uint8 idx = roundAddrToIdx[roundId][trainer];
            require(idx != 0, "Trainer not mapped");

            uint8 nPlayers = roundTrainerCount[roundId];
            require(nPlayers > 0, "Invalid player count");

            uint256 idxMask = 1 << (idx - 1);
            value = _calcShapleyValuePerm(roundId, idxMask, nPlayers);
        }

        isPositive = value > 0;
        if (!isPositive) value = -value;
    }

    /**
     * @notice Estimate gas cost for computing Shapley value
     * @param roundId Round ID
     * @param trainer Trainer address
     * @return gasEstimate Estimated gas cost
     */
    function estimateShapleyGas(
        uint256 roundId,
        address trainer
    ) external view returns (uint256 gasEstimate) {
        uint8 idx = roundAddrToIdx[roundId][trainer];
        require(idx != 0, "Trainer not mapped");

        // Base cost + cost per sample
        // These are rough estimates, adjust based on testing
        uint256 baseCost = 50000;
        uint256 costPerSample = 3000;

        gasEstimate = baseCost + (SAMPLES_PER_PLAYER * costPerSample);
    }

    function recoverERC20(address token, uint256 amount) external onlyOwner {
        IERC20(token).safeTransfer(owner(), amount);
    }

    /** TODO: remove this: only for testing purposes */
    function setShapleyCoalitionValues(
        uint256 roundId,
        uint256[][] calldata coalitionIndices,
        uint256[] calldata values
    ) external onlyOwner {
        require(coalitionIndices.length == values.length, "Length mismatch");

        uint8 n = roundTrainerCount[roundId];
        require(n > 0, "Round mapping not revealed");

        for (uint256 i = 0; i < coalitionIndices.length; ++i) {
            uint256 mask = _idxArrayToMask(coalitionIndices[i]);

            bytes32 cid = keccak256(abi.encodePacked(roundId, mask));

            Coalition storage c = coalitionData[cid];

            if (!c.committed) {
                c.committed = true;
                c.revealed = true;
                c.bitfield = bytes32(mask);
                c.nonce = bytes32(0);
                c.numScores = 0;
                c.sumScores = 0;
                roundMaskToCid[roundId][mask] = cid;
                roundCoalitionExists[roundId][mask] = true;
                roundAllCoalitions[roundId].push(mask);
            }

            c.result = values[i];

            roundCoalitionValues[roundId][mask] = values[i];

            emit ShapleyCoalitionValueSet(roundId, mask, values[i]);
        }
    }

    function _idxArrayToMask(
        uint256[] calldata idxs
    ) private pure returns (uint256 m) {
        for (uint256 i; i < idxs.length; i++) {
            require(idxs[i] > 0 && idxs[i] <= 255, "idx out");
            uint256 bit = 1 << (idxs[i] - 1);
            require(m & bit == 0, "dup");
            m |= bit;
        }
    }

    function _popcnt(uint256 x) private pure returns (uint8 c) {
        while (x != 0) {
            x &= x - 1;
            unchecked {
                ++c;
            }
        }
    }
}
