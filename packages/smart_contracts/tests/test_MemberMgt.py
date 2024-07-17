import ape
import pytest
from eth_tester.exceptions import TransactionFailed
from solcx import compile_standard
import json
from eth_hash.auto import keccak
from eth_utils import to_bytes
@pytest.fixture
def accounts_fixture(accounts):
    return accounts[:4]  # Use first four accounts

@pytest.fixture
def account(accounts_fixture):
    return accounts_fixture[0]

@pytest.fixture
def new_member(accounts_fixture):
    return accounts_fixture[3]

@pytest.fixture
def members(accounts_fixture):
    return accounts_fixture[:3]  # First three accounts as initial members

@pytest.fixture
def new_model_data():
    model_data = {
        "name": "Model1",
        "parameters": {
            "param1": 0.1,
            "param2": 100,
            "param3": "abc"
        }
    }
    # Serialize model_data to JSON and encode as bytes
    model_data_bytes = json.dumps(model_data).encode('utf-8')
    
    # Calculate the keccak256 hash of the model_data_bytes
    modelHash = keccak(model_data_bytes)
    
    return modelHash

@pytest.fixture
def memberManagementContract(account, project, members):
    return account.deploy(project.MemberManagement, members, 3)

def test_proposeAddMember(account, new_member, memberManagementContract):
    # Arrange
    proposalCount = memberManagementContract.proposalCount()
    
    # Act
    memberManagementContract.proposeAddMember(new_member, sender=account)
    
    # Assert
    proposal = memberManagementContract.proposals(proposalCount)
    assert proposal.proposer == account
    assert proposal.member == new_member
    assert proposal.add is True
    assert proposal.signatures == 0
    proposalCountIncrement = memberManagementContract.proposalCount()
    assert proposalCount + 1 == proposalCountIncrement

def test_proposeRemoveMember(account, new_member, memberManagementContract, members):
    # Arrange
    # Propose to add a new member first, to ensure there's a member to remove
    memberManagementContract.proposeAddMember(new_member, sender=account)
    proposal_id_add = memberManagementContract.proposalCount() - 1
    for member in members:
        memberManagementContract.signProposal(proposal_id_add, sender=member)

    # Act
    proposalCount = memberManagementContract.proposalCount()
    memberManagementContract.proposeRemoveMember(new_member, sender=account)
    
    # Assert
    proposal = memberManagementContract.proposals(proposalCount)
    assert proposal.proposer == account
    assert proposal.member == new_member
    assert proposal.add is False
    assert proposal.signatures == 0
    proposalCountIncrement = memberManagementContract.proposalCount()
    assert proposalCount + 1 == proposalCountIncrement

def test_signProposal(account, new_member, memberManagementContract, members):
    # Arrange
    memberManagementContract.proposeAddMember(new_member, sender=account)
    proposal_id = memberManagementContract.proposalCount() - 1
    
    # Act & Assert
    for member in members:
        memberManagementContract.signProposal(proposal_id, sender=member)
        proposal = memberManagementContract.proposals(proposal_id)
        assert memberManagementContract.signatures(proposal_id, member) is True, "member signed"
        assert proposal.signatures == members.index(member) + 1

    # Check that member is added after threshold is met
    assert memberManagementContract.isMember(new_member) is True, "is members "
    assert memberManagementContract.isWhitelisted(new_member) is True, "is whitelisted"

def test_isWhitelisted(account, new_member, memberManagementContract, members):
    # Arrange
    # New member not whitelisted yet
    assert memberManagementContract.isWhitelisted(new_member) is False
    
    # Act
    memberManagementContract.proposeAddMember(new_member, sender=account)
    proposal_id = memberManagementContract.proposalCount() - 1
    for member in members:
        memberManagementContract.signProposal(proposal_id, sender=member)

    # Assert
    assert memberManagementContract.isWhitelisted(new_member) is True

def test_getMembers(memberManagementContract, members):
    # Act
    contract_members = memberManagementContract.getMembers()
    
    # Assert
    assert contract_members == members

def test_getMemberStatus(account, new_member, memberManagementContract, members):
    # Arrange
    # Initially, new_member is neither a member nor whitelisted
    member_status = memberManagementContract.getMemberStatus(new_member)
    assert member_status == (False, False)
    
    # Act
    memberManagementContract.proposeAddMember(new_member, sender=account)
    proposal_id = memberManagementContract.proposalCount() - 1
    for member in members:
        memberManagementContract.signProposal(proposal_id, sender=member)
    
    # Assert
    member_status = memberManagementContract.getMemberStatus(new_member)
    assert member_status == (True, True)

def test_submitModelUpdate(account, new_model_data, memberManagementContract):
    # Arrange
    # Generate a dummy signature for the model update (replace with valid signature)
    signature = b'\x00' * 65
    
    initial_model_updates_count = memberManagementContract.modelCount()
    
    # Act
    memberManagementContract.submitModelUpdate(new_model_data, signature, sender=account)
    
    # Assert
    # Check if the ModelUpdateSubmitted event was emitted (if possible)
    # Example assertion if you have access to emitted events
    # assert len(memberManagementContract.events.ModelUpdateSubmitted) == initial_model_updates_count + 1
    
    # Check if the model update was stored correctly in the contract state
    model_update = memberManagementContract.modelUpdates(new_model_data)
    assert model_update[0] == new_model_data
    assert model_update[1] == account
    assert model_update[2] == False  # Assuming verified flag starts as false
    
    # Additional checks based on your contract's logic
    # Verify that only whitelisted members can submit model updates
    assert memberManagementContract.isWhitelisted(account)
    
    # Verify that non-whitelisted members cannot submit model updates
    with pytest.raises(TransactionFailed):
        memberManagementContract.submitModelUpdate(new_model_data, signature, sender=other_account)

#################################

""" def test_submitModelUpdate(account, memberManagementContract):
    # Arrange
    modelHash = "0x" + "1" * 64  # Replace with actual model hash
    signature = account.sign_message(modelHash)

    # Act
    memberManagementContract.submitModelUpdate(modelHash, signature, {"from": account})

    # Assert
    modelUpdate = memberManagementContract.modelUpdates(modelHash)
    assert modelUpdate.modelHash == modelHash
    assert modelUpdate.signer == account
    assert modelUpdate.verified is False

def test_verifyModelUpdate(account, memberManagementContract):
    # Arrange
    modelHash = "0x" + "1" * 64  # Replace with actual model hash
    signature = account.sign_message(modelHash)
    memberManagementContract.submitModelUpdate(modelHash, signature, {"from": account})

    # Act
    memberManagementContract.verifyModelUpdate(modelHash, {"from": account})

    # Assert
    modelUpdate = memberManagementContract.modelUpdates(modelHash)
    assert modelUpdate.verified is True

def test_addModel(account, memberManagementContract):
    # Arrange
    ipfsHash = "QmTmVJYNrRTedkkfns4u6NYzM1VN1KHF9nGz4SW6W8PtBy"

    # Act
    memberManagementContract.addModel(ipfsHash, {"from": account})

    # Assert
    model = memberManagementContract.clientHistory(account, memberManagementContract.round())
    assert model.owner == account
    assert model.ipfsHash == ipfsHash
    assert model.timestamp > 0

def test_getModelIPFSHash(account, memberManagementContract):
    # Arrange
    ipfsHash = "QmTmVJYNrRTedkkfns4u6NYzM1VN1KHF9nGz4SW6W8PtBy"
    memberManagementContract.addModel(ipfsHash, {"from": account})

    # Act
    storedIPFSHash = memberManagementContract.getModelIPFSHash(account, memberManagementContract.round())

    # Assert
    assert storedIPFSHash == ipfsHash

def test_startNewRound(account, memberManagementContract):
    # Arrange
    initialRound = memberManagementContract.round()

    # Act
    memberManagementContract.startNewRound({"from": account})

    # Assert
    newRound = memberManagementContract.round()
    assert newRound == initialRound + 1
    assert memberManagementContract.proposalCount() == 0

 """
##########################

""" import ape
import pytest
import hashlib
from eth_account.messages import encode_defunct

@pytest.fixture
def accounts_fixture(accounts):
    return accounts[:4]  # Use first four accounts

@pytest.fixture
def account(accounts_fixture):
    return accounts_fixture[0]

@pytest.fixture
def new_member(accounts_fixture):
    return accounts_fixture[3]

@pytest.fixture
def members(accounts_fixture):
    return accounts_fixture[:3]  # First three accounts as initial members

@pytest.fixture
def memberManagementContract(account, project, members):
    return account.deploy(project.MemberManagement, members, 3)

@pytest.fixture
def model_hash():
    return hashlib.sha256(b"model data").digest()

@pytest.fixture
def signature(account, model_hash):
    message = encode_defunct(text=hashlib.sha256(model_hash + account.address.encode()).hexdigest())
    signed_message = account.sign_message(message)

    # Ensure each component of the signature is in bytes
    v = signed_message.v if isinstance(signed_message.v, bytes) else signed_message.v.to_bytes(1, byteorder='big')
    r = signed_message.r if isinstance(signed_message.r, bytes) else signed_message.r.to_bytes(32, byteorder='big')
    s = signed_message.s if isinstance(signed_message.s, bytes) else signed_message.s.to_bytes(32, byteorder='big')

    signature = v + r + s
    print(f"Signed message: {signed_message}")
    print(f"Signature: {signature.hex()}")
    return signature

def test_submitModelUpdate(account, memberManagementContract, model_hash, signature):
    # Arrange
    model_hash_hex = "0x" + model_hash.hex()

    # Act
    memberManagementContract.submitModelUpdate(model_hash_hex, signature, sender=account)
    
    # Assert
    model_update = memberManagementContract.modelUpdates(model_hash_hex)
    assert model_update.modelHash == model_hash
    assert model_update.signer == account.address
    assert model_update.verified is False

def test_verifyModelUpdate(account, memberManagementContract, model_hash, signature):
    # Arrange
    model_hash_hex = "0x" + model_hash.hex()
    memberManagementContract.submitModelUpdate(model_hash_hex, signature, sender=account)
    
    # Act
    memberManagementContract.verifyModelUpdate(model_hash_hex, sender=account)
    
    # Assert
    model_update = memberManagementContract.modelUpdates(model_hash_hex)
    assert model_update.modelHash == model_hash
    assert model_update.signer == account.address
    assert model_update.verified is True """
