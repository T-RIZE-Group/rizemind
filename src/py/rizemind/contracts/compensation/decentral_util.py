from flwr.common.typing import Parameters


def encode_parameters(parameters: Parameters) -> dict:
    parameters_dict = parameters.__dict__
    parameters_dict["tensors"] = [str(tensor) for tensor in parameters_dict["tensors"]]
    return parameters_dict


def decode_parameters(parameters_dict: dict) -> Parameters:
    tensor_type = parameters_dict["tensor_type"]
    tensors = [bytes(tensor) for tensor in parameters_dict["tensors"]]
    return Parameters(tensors, tensor_type)  # type: ignore
