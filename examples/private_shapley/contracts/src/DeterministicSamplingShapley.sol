// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title ImprovedPrivateShapley – Dynamic trainer mapping with on-chain Shapley
 * @author Blockchain Dev Team
 */
contract ImprovedPrivateShapley is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    /*──────────────────────────────────────────────────────────────────
                                CONSTANTS
    ──────────────────────────────────────────────────────────────────*/
    uint8 public constant MAX_TRAINERS = 255;
    uint8 public constant MAX_SHAPLEY_PLAYERS = 20;
    uint256 public constant PRECISION = 1e6;
    uint256 public constant MAX_BATCH_SIZE = 256;
    uint256 public constant COMMIT_REVEAL_WINDOW = 7 days;
    uint256 public constant MIN_RESULT_THRESHOLD = 1;
    uint8 public constant NUM_SAMPLES = 128;
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

    /**
     * this is good
     */
    function commitTrainerMapping(
        uint256 rid,
        bytes32 commitment
    ) external onlyOwner {
        require(mappingCommit[rid].commitment == 0, "already committed");
        mappingCommit[rid].commitment = commitment;
        emit MappingCommitted(rid, commitment);
    }

    /**
     * Why the salt option instead of the merkle tree?
     * You won't even need this transaction, you can
     * simply reveal the proof to trainers offchain.
     */
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
              3.  ON-CHAIN SHAPLEY VALUE  (exact for ≤20 players)
    ──────────────────────────────────────────────────────────────────*/
    mapping(uint256 => mapping(uint256 => uint256)) public roundCoalitionValues; // round => mask => value
    mapping(uint256 => mapping(uint256 => bool)) public roundCoalitionExists;
    mapping(uint256 => uint256[]) private roundAllCoalitions;

    event ShapleyCoalitionValueSet(
        uint256 roundId,
        uint256 mask,
        uint256 value
    );

    /* ---------- Fisher‑Yates shuffle derived from a public seed ---------- */
    function _permutation(
        uint256 seed,
        uint8 n
    ) internal pure returns (uint8[] memory order) {
        order = new uint8[](n);
        for (uint8 i = 0; i < n; ++i) order[i] = i; // 0‑based indices

        bytes32 h = keccak256(abi.encodePacked(seed));
        for (uint8 i = n; i > 1; --i) {
            uint256 j = uint256(h) % i; // 0 ≤ j < i
            (order[i - 1], order[j]) = (order[j], order[i - 1]);
            h = keccak256(abi.encodePacked(h)); // advance hash
        }
    }

    /* ---------- DP‑Shapley estimator ---------- */
    function _calcDPShapley(
        uint256 roundId,
        uint256 trainerBit, // single‑bit mask of the target trainer
        uint8 nPlayers
    ) internal view returns (int256 est) {
        for (uint8 k = 0; k < NUM_SAMPLES; ++k) {
            uint8[] memory perm = _permutation(
                uint256(keccak256(abi.encodePacked(roundId, k))), // deterministic seed
                nPlayers
            );

            uint256 prefixMask = 0;
            for (uint8 p = 0; p < perm.length; ++p) {
                uint256 plBit = 1 << perm[p]; // 1‑based → 0‑based mapping
                if (plBit == trainerBit) {
                    // v(S ∪ {i}) − v(S)
                    bytes32 cidWith = roundMaskToCid[roundId][
                        prefixMask | trainerBit
                    ];
                    bytes32 cidWithout = roundMaskToCid[roundId][prefixMask];

                    int256 diff = int256(coalitionData[cidWith].result) -
                        int256(coalitionData[cidWithout].result);
                    est += diff;
                    break;
                }
                prefixMask |= plBit;
            }
        }
        est /= int256(uint256(NUM_SAMPLES)); // arithmetic mean over K permutations
    }

    /*──────────────────────────────────────────────────────────────────
                              4.  CLAIM REWARDS
    ──────────────────────────────────────────────────────────────────*/
    mapping(uint256 => mapping(address => bool)) public roundRewardClaimed; // round ⇒ trainer ⇒ claimed?

    event RewardClaimed(address trainer, bytes32 coalition, uint256 amount);

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

        int256 shapley = _calcDPShapley(rid, idxMask, nPlayers);
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
        uint8 idx = roundAddrToIdx[roundId][trainer];
        require(idx != 0, "Trainer not mapped");

        uint8 nPlayers = roundTrainerCount[roundId];
        require(
            nPlayers > 0 && nPlayers <= MAX_SHAPLEY_PLAYERS,
            "Invalid player count"
        );

        uint256 idxMask = 1 << (idx - 1);
        value = _calcDPShapley(roundId, idxMask, nPlayers);
        isPositive = value > 0;
        if (!isPositive) value = -value;
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
        require(n <= MAX_SHAPLEY_PLAYERS, "Too many trainers");

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
}
