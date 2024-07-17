
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

contract MemberManagement {
    address[] public members;
    mapping(address => bool) public isMember;
    mapping(uint256 => Proposal) public proposals;
    mapping(address => bool) public whitelist;
    uint256 public proposalCount;
    uint256 public threshold;
    uint256 public round; // Define round as a public state variable
    mapping(uint256 => mapping(address => bool)) public signatures;
    mapping(string => bool) public existIPFS; // Mapping to store IPFS hashes

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

    event ProposalCreated(uint256 proposalId, address proposer, address member, bool add);
    event ProposalSigned(uint256 proposalId, address signer);
    event MemberAdded(address member);
    event MemberRemoved(address member);
    event ModelUpdateSubmitted(bytes32 modelHash, address signer);
    event ModelUpdateVerified(bytes32 modelHash, address signer);
    event ModelAdded(address indexed owner, uint256 indexed round, string ipfsHash);
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

    constructor(address[] memory initialMembers, uint256 initialThreshold) {
        for (uint256 i = 0; i < initialMembers.length; i++) {
            members.push(initialMembers[i]);
            isMember[initialMembers[i]] = true;
            whitelist[initialMembers[i]] = true;
        }
        threshold = initialThreshold;
        round = 0; // Initialize round number
    }

    function proposeAddMember(address member) public onlyMember {
        proposals[proposalCount] = Proposal({
            proposer: msg.sender,
            member: member,
            add: true,
            signatures: 0
        });
        emit ProposalCreated(proposalCount, msg.sender, member, true);
        proposalCount++;
    }

    function proposeRemoveMember(address member) public onlyMember {
        proposals[proposalCount] = Proposal({
            proposer: msg.sender,
            member: member,
            add: false,
            signatures: 0
        });
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

    function getMemberStatus(address _address) public view returns (bool member, bool whitelisted) {
        return (isMember[_address], whitelist[_address]);
    }

    function submitModelUpdate(bytes32 modelHash, bytes memory signature) public onlyMember {
        require(verifySignature(modelHash, msg.sender, signature), "Invalid signature");
        require(whitelist[msg.sender], "Signer is not whitelisted");

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
        update.verified = true;
        emit ModelUpdateVerified(modelHash, update.signer);
    }

    function addModel(string memory _ipfsHash)
        public
        onlyWhitelisted
        onlyOnce(_ipfsHash)
    {
        existIPFS[_ipfsHash] = true;

        Model memory newModel = Model({
            owner: msg.sender,
            round: round, // Update with actual round number as needed
            timestamp: block.timestamp,
            ipfsHash: _ipfsHash
        });

        clientHistory[msg.sender][round] = newModel; // Update with actual round number as needed
        emit ModelAdded(msg.sender, round, _ipfsHash); // Update with actual round number as needed
    }

    function verifySignature(bytes32 modelHash, address signer, bytes memory signature) internal pure returns (bool) {
        bytes32 message = prefixed(keccak256(abi.encodePacked(modelHash, signer)));
        return (recoverSigner(message, signature) == signer);
    }

    function prefixed(bytes32 hash) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash));
    }

    function recoverSigner(bytes32 message, bytes memory sig) internal pure returns (address) {
        uint8 v;
        bytes32 r;
        bytes32 s;
        (v, r, s) = splitSignature(sig);
        return ecrecover(message, v, r, s);
    }

    function splitSignature(bytes memory sig) internal pure returns (uint8, bytes32, bytes32) {
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

    // Function to retrieve the IPFS hash associated with a member's model in a specific round
    function getModelIPFSHash(address member, uint256 _round) public view returns (string memory) {
        return clientHistory[member][_round].ipfsHash;
    }

    // Function to start a new round
    function startNewRound() public onlyMember {
        round++;
        proposalCount = 0; // Reset proposal count for the new round
        emit NewRoundStarted(round);
    }
}

/* pragma solidity ^0.8.10;

contract MemberManagement {
    address[] public members;
    mapping(address => bool) public isMember;
    mapping(uint256 => Proposal) public proposals;
    mapping(address => bool) public whitelist;
    uint256 public proposalCount;
    uint256 public threshold;
    uint256 public round; // Define round as a public state variable
    mapping(uint256 => mapping(address => bool)) public signatures;
    mapping(string => bool) public existIPFS; // Mapping to store IPFS hashes

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

    event ProposalCreated(uint256 proposalId, address proposer, address member, bool add);
    event ProposalSigned(uint256 proposalId, address signer);
    event MemberAdded(address member);
    event MemberRemoved(address member);
    event ModelUpdateSubmitted(bytes32 modelHash, address signer);
    event ModelUpdateVerified(bytes32 modelHash, address signer);
    event ModelAdded(address indexed owner, uint256 indexed round, string ipfsHash);

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

    constructor(address[] memory initialMembers, uint256 initialThreshold) {
        for (uint256 i = 0; i < initialMembers.length; i++) {
            members.push(initialMembers[i]);
            isMember[initialMembers[i]] = true;
            whitelist[initialMembers[i]] = true;
        }
        threshold = initialThreshold;
        round = 0; // Initialize round number
    }

    function proposeAddMember(address member) public onlyMember {
        proposals[proposalCount] = Proposal({
            proposer: msg.sender,
            member: member,
            add: true,
            signatures: 0
        });
        emit ProposalCreated(proposalCount, msg.sender, member, true);
        proposalCount++;
    }

    function proposeRemoveMember(address member) public onlyMember {
        proposals[proposalCount] = Proposal({
            proposer: msg.sender,
            member: member,
            add: false,
            signatures: 0
        });
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

    function getMemberStatus(address _address) public view returns (bool member, bool whitelisted) {
        return (isMember[_address], whitelist[_address]);
    }

    function submitModelUpdate(bytes32 modelHash, bytes memory signature) public onlyMember {
        require(verifySignature(modelHash, msg.sender, signature), "Invalid signature");
        require(whitelist[msg.sender], "Signer is not whitelisted");

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
        update.verified = true;
        emit ModelUpdateVerified(modelHash, update.signer);
    }

    function addModel(string memory _ipfsHash)
        public
        onlyWhitelisted
        onlyOnce(_ipfsHash)
    {
        existIPFS[_ipfsHash] = true;

        Model memory newModel = Model({
            owner: msg.sender,
            round: round, // Update with actual round number as needed
            timestamp: block.timestamp,
            ipfsHash: _ipfsHash
        });

        clientHistory[msg.sender][round] = newModel; // Update with actual round number as needed
        emit ModelAdded(msg.sender, round, _ipfsHash); // Update with actual round number as needed
    }

    function verifySignature(bytes32 modelHash, address signer, bytes memory signature) internal pure returns (bool) {
        bytes32 message = prefixed(keccak256(abi.encodePacked(modelHash, signer)));
        return (recoverSigner(message, signature) == signer);
    }

    function prefixed(bytes32 hash) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash));
    }

    function recoverSigner(bytes32 message, bytes memory sig) internal pure returns (address) {
        uint8 v;
        bytes32 r;
        bytes32 s;
        (v, r, s) = splitSignature(sig);
        return ecrecover(message, v, r, s);
    }

    function splitSignature(bytes memory sig) internal pure returns (uint8, bytes32, bytes32) {
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

    // Function to retrieve the IPFS hash associated with a member's model in a specific round
    function getModelIPFSHash(address member, uint256 _round) public view returns (string memory) {
        return clientHistory[member][_round].ipfsHash;
    }
} */


/* pragma solidity ^0.8.10;

contract MemberManagement {
    address[] public members;
    mapping(address => bool) public isMember;
    mapping(uint256 => Proposal) public proposals;
    mapping(address => bool) public whitelist;
    uint256 public proposalCount;
    uint256 public threshold;
    mapping(uint => mapping(address => bool)) public signatures;

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

    mapping(bytes32 => ModelUpdate) public modelUpdates;

    event ProposalCreated(uint256 proposalId, address proposer, address member, bool add);
    event ProposalSigned(uint256 proposalId, address signer);
    event MemberAdded(address member);
    event MemberRemoved(address member);
    event ModelUpdateSubmitted(bytes32 modelHash, address signer);
    event ModelUpdateVerified(bytes32 modelHash, address signer);

    modifier onlyMember() {
        require(isMember[msg.sender], "Not a member");
        _;
    }

    constructor(address[] memory initialMembers, uint256 initialThreshold) {
        for (uint256 i = 0; i < initialMembers.length; i++) {
            members.push(initialMembers[i]);
            isMember[initialMembers[i]] = true;
            whitelist[initialMembers[i]] = true;
        }
        threshold = initialThreshold;
    }

    function proposeAddMember(address member) public onlyMember {
        proposals[proposalCount] = Proposal({
            proposer: msg.sender,
            member: member,
            add: true,
            signatures: 0
        });
        emit ProposalCreated(proposalCount, msg.sender, member, true);
        proposalCount++;
    }

    function proposeRemoveMember(address member) public onlyMember {
        proposals[proposalCount] = Proposal({
            proposer: msg.sender,
            member: member,
            add: false,
            signatures: 0
        });
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

    function getMemberStatus(address _address) public view returns (bool member, bool whitelisted) {
        return (isMember[_address], whitelist[_address]);
    }

    function submitModelUpdate(bytes32 modelHash, bytes memory signature) public onlyMember {
        require(verifySignature(modelHash, msg.sender, signature), "Invalid signature");
        require(whitelist[msg.sender], "Signer is not whitelisted");
        
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
        update.verified = true;
        emit ModelUpdateVerified(modelHash, update.signer);
    }

    function verifySignature(bytes32 modelHash, address signer, bytes memory signature) internal pure returns (bool) {
        bytes32 message = prefixed(keccak256(abi.encodePacked(modelHash, signer)));
        return (recoverSigner(message, signature) == signer);
    }

    function prefixed(bytes32 hash) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash));
    }

    function recoverSigner(bytes32 message, bytes memory sig) internal pure returns (address) {
        uint8 v;
        bytes32 r;
        bytes32 s;
        (v, r, s) = splitSignature(sig);
        return ecrecover(message, v, r, s);
    }

    function splitSignature(bytes memory sig) internal pure returns (uint8, bytes32, bytes32) {
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
}  */

/* // 

contract MemberManagement {
    address[] public members;
    mapping(address => bool) public isMember;
    mapping(uint256 => Proposal) public proposals;
    mapping(address => bool) public whitelist;
    uint256 public proposalCount;
    uint256 public threshold;
    mapping(uint => mapping(address => bool)) public signatures;

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

    mapping(bytes32 => ModelUpdate) public modelUpdates;

    event ProposalCreated(uint256 proposalId, address proposer, address member, bool add);
    event ProposalSigned(uint256 proposalId, address signer);
    event MemberAdded(address member);
    event MemberRemoved(address member);
    event ModelUpdateSubmitted(bytes32 modelHash, address signer);
    event ModelUpdateVerified(bytes32 modelHash, address signer);

    modifier onlyMember() {
        require(isMember[msg.sender], "Not a member");
        _;
    }

    constructor(address[] memory initialMembers, uint256 initialThreshold) {
        for (uint256 i = 0; i < initialMembers.length; i++) {
            members.push(initialMembers[i]);
            isMember[initialMembers[i]] = true;
            whitelist[initialMembers[i]] = true;
        }
        threshold = initialThreshold;
    }

    function proposeAddMember(address member) public onlyMember {
        proposals[proposalCount] = Proposal({
            proposer: msg.sender,
            member: member,
            add: true,
            signatures: 0
        });
        emit ProposalCreated(proposalCount, msg.sender, member, true);
        proposalCount++;
    }

    function proposeRemoveMember(address member) public onlyMember {
        proposals[proposalCount] = Proposal({
            proposer: msg.sender,
            member: member,
            add: false,
            signatures: 0
        });
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

    function getMemberStatus(address _address) public view returns (bool member, bool whitelisted) {
        return (isMember[_address], whitelist[_address]);
    }

    function submitModelUpdate(bytes32 modelHash) public onlyMember {
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
        update.verified = true;
        emit ModelUpdateVerified(modelHash, update.signer);
    }
}
 */

