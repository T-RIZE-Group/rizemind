// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

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

    event ProposalCreated(uint256 proposalId, address proposer, address member, bool add);
    event ProposalSigned(uint256 proposalId, address signer);
    event MemberAdded(address member);
    event MemberRemoved(address member);

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
}
