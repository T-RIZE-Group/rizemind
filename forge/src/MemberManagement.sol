// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MemberManagement {
    string private _name;

    address[] public members;
    mapping(address => bool) public isMember;
    mapping(uint256 => Proposal) public proposals;
    mapping(address => bool) public whitelist;

    uint256 public proposalCount;
    uint256 public modelCount; // State variable for model count
    uint256 public threshold;
    uint256 public round; // Define round as a public state variable
    mapping(uint256 => mapping(address => bool)) public signatures;
    mapping(string => bool) public existIPFS; // Mapping to store IPFS hashes
    // A mapping to store the submitted models with their hashes
    mapping(bytes32 => bool) public submittedModels;

    struct Proposal {
        address proposer;
        address member;
        bool add;
        uint256 signatures;
    }

    struct ModelUpdate {
        bytes32 modelHash;
        address signer;
        bool verified;
    }

    struct Model {
        address owner;
        uint256 round; // Assuming this is the round number
        uint256 timestamp;
        string ipfsHash;
    }

    mapping(bytes32 => ModelUpdate) public modelUpdates;
    mapping(address => mapping(uint256 => Model)) public clientHistory;

    event ProposalCreated(
        uint256 proposalId,
        address proposer,
        address member,
        bool add
    );
    event ProposalSigned(uint256 proposalId, address signer);
    event MemberAdded(address member);
    event MemberRemoved(address member);
    event ModelUpdateSubmitted(bytes32 modelHash, address signer);
    event ModelUpdateVerified(bytes32 modelHash, address signer);
    event ModelAdded(
        address indexed owner,
        uint256 indexed round,
        string ipfsHash
    );
    event NewRoundStarted(uint256 newRound);

    modifier onlyMember() {
        require(isMember[msg.sender], "Not a member");
        _;
    }

    modifier onlyWhitelisted() {
        require(whitelist[msg.sender], "Not whitelisted");
        _;
    }

    modifier onlyOnce(string memory _ipfsHash) {
        require(!existIPFS[_ipfsHash], "IPFS hash already exists");
        _;
    }

    constructor(
        string memory name_,
        address[] memory initialMembers,
        uint256 initialThreshold
    ) {
        _name = name_;
        for (uint256 i = 0; i < initialMembers.length; i++) {
            members.push(initialMembers[i]);
            isMember[initialMembers[i]] = true;
            whitelist[initialMembers[i]] = true;
        }
        threshold = initialThreshold;
        round = 0; // Initialize round number
    }

    function name() public view returns (string memory) {
        return _name;
    }

    function proposeAddMember(address member) public onlyMember {
        proposals[proposalCount] = Proposal({
            proposer: msg.sender,
            member: member,
            add: true,
            signatures: 0
        });
        signProposal(proposalCount);
        emit ProposalCreated(proposalCount, msg.sender, member, true);
        proposalCount++;
    }

    function proposeRemoveMember(address member) public onlyMember {
        require(isMember[member], "not a member");
        proposals[proposalCount] = Proposal({
            proposer: msg.sender,
            member: member,
            add: false,
            signatures: 0
        });
        signProposal(proposalCount);
        emit ProposalCreated(proposalCount, msg.sender, member, false);
        proposalCount++;
    }

    function signProposal(uint256 proposalId) public onlyMember {
        Proposal storage proposal = proposals[proposalId];
        require(!signatures[proposalId][msg.sender], "Already signed");
        signatures[proposalId][msg.sender] = true;
        proposal.signatures++;

        if (proposal.signatures >= threshold) {
            if (proposal.add) {
                members.push(proposal.member);
                isMember[proposal.member] = true;
                whitelist[proposal.member] = true;
                emit MemberAdded(proposal.member);
            } else {
                for (uint256 i = 0; i < members.length; i++) {
                    if (members[i] == proposal.member) {
                        members[i] = members[members.length - 1];
                        members.pop();
                        isMember[proposal.member] = false;
                        whitelist[proposal.member] = false;
                        emit MemberRemoved(proposal.member);
                        break;
                    }
                }
            }
        }

        emit ProposalSigned(proposalId, msg.sender);
    }

    function isWhitelisted(address _address) public view returns (bool) {
        return whitelist[_address];
    }

    function getMembers() public view returns (address[] memory) {
        return members;
    }

    function getMemberStatus(
        address _address
    ) public view returns (bool member, bool whitelisted) {
        return (isMember[_address], whitelist[_address]);
    }

    function getModelCount() public view returns (uint256) {
        return modelCount;
    }

    function submitModelUpdate(
        bytes memory newModelData,
        bytes memory signature
    ) public {
        require(isValidSignature(newModelData, signature), "Invalid signature");

        bytes32 modelHash = keccak256(newModelData);
        require(!submittedModels[modelHash], "Model already submitted");

        // Mark the model as submitted
        submittedModels[modelHash] = true;
        modelCount += 1;

        // Store the model update
        modelUpdates[modelHash] = ModelUpdate({
            modelHash: modelHash,
            signer: msg.sender,
            verified: false
        });

        emit ModelUpdateSubmitted(modelHash, msg.sender);
    }

    function verifyModelUpdate(bytes32 modelHash) public onlyMember {
        ModelUpdate storage update = modelUpdates[modelHash];
        require(update.signer != address(0), "Model update does not exist");
        require(whitelist[update.signer], "Signer is not whitelisted");

        // Mark the model update as verified
        update.verified = true;

        emit ModelUpdateVerified(modelHash, update.signer);
    }

    function addModel(
        string memory _ipfsHash
    ) public onlyWhitelisted onlyOnce(_ipfsHash) {
        existIPFS[_ipfsHash] = true;

        Model memory newModel = Model({
            owner: msg.sender,
            round: round,
            timestamp: block.timestamp,
            ipfsHash: _ipfsHash
        });

        clientHistory[msg.sender][round] = newModel;
        emit ModelAdded(msg.sender, round, _ipfsHash);
    }

    function isValidSignature(
        bytes memory data,
        bytes memory signature
    ) internal view returns (bool) {
        bytes32 messageHash = keccak256(data);
        bytes32 prefixedHash = prefixed(messageHash);
        address recoveredAddress = recoverSigner(prefixedHash, signature);
        return whitelist[recoveredAddress];
    }

    function prefixed(bytes32 hash) internal pure returns (bytes32) {
        return
            keccak256(
                abi.encodePacked("\x19Ethereum Signed Message:\n32", hash)
            );
    }

    function recoverSigner(
        bytes32 message,
        bytes memory sig
    ) internal pure returns (address) {
        uint8 v;
        bytes32 r;
        bytes32 s;
        (v, r, s) = splitSignature(sig);
        return ecrecover(message, v, r, s);
    }

    function splitSignature(
        bytes memory sig
    ) internal pure returns (uint8, bytes32, bytes32) {
        require(sig.length == 65, "invalid signature length");
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
        return (v, r, s);
    }

    function getModelIPFSHash(
        address member,
        uint256 _round
    ) public view returns (string memory) {
        return clientHistory[member][_round].ipfsHash;
    }

    function startNewRound() public onlyMember {
        round++;
        proposalCount = 0; // Reset proposal count for the new round
        emit NewRoundStarted(round);
    }
}
