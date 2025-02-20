from pathlib import Path
from solcx import compile_source, install_solc

####
# POC code, should not be use
# use factories instead
####


class CompiledContract:
    name: str
    abi: str
    bytecode: str

    def __init__(self, name: str, abi: str, bytecode: str):
        self.name = name
        self.abi = abi
        self.bytecode = bytecode

    def __repr__(self):
        return f"CompiledContract(abi={self.abi[:30]}..., bytecode={self.bytecode[:30]}...)"


def load_contract(contract_file_name: str) -> str:
    # Get the directory of the current Python file
    current_dir = Path(__file__).parent

    # Construct the path to the Solidity file
    file_path = (
        current_dir / f"../../../../smart_contracts/contracts/{contract_file_name}"
    )
    file_path = file_path.resolve()  # Resolve to an absolute path
    # Read the Solidity file
    with open(file_path, "r") as file:
        contract_source_code = file.read()

    return contract_source_code


def compile_contract(src: str, contract_name: str) -> CompiledContract:
    version = install_solc(show_progress=True)
    print(f"installed solc {version}")
    compiled_sol = compile_source(src, solc_version=version)
    contract = compiled_sol[f"<stdin>:{contract_name}"]
    return CompiledContract(contract_name, contract["abi"], contract["bin"])


def load_compile(file_name: str, contract_name: str) -> CompiledContract:
    source = load_contract(file_name)
    return compile_contract(source, contract_name)
