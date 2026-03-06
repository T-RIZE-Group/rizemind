"""Serialization of Flower Parameters and FitRes for TEE transport.

Provides compact binary packing so model data can be encrypted as a single
byte buffer and sent over vsock to the Nitro Enclave.

Wire format (Parameters)::

    [tensor_type_len: 4B][tensor_type: N bytes]
    [n_tensors: 4B]
    [tensor_0_len: 4B][tensor_0_bytes] ...

Wire format (FitRes — wraps Parameters)::

    [num_examples: 4B]
    [parameters: ...]
"""

import struct

from flwr.common.typing import Parameters


def serialize_parameters(parameters: Parameters) -> bytes:
    """Serialize Flower ``Parameters`` into a single byte buffer."""
    parts: list[bytes] = []
    tt = parameters.tensor_type.encode("utf-8")
    parts.append(struct.pack("!I", len(tt)))
    parts.append(tt)
    parts.append(struct.pack("!I", len(parameters.tensors)))
    for tensor in parameters.tensors:
        parts.append(struct.pack("!I", len(tensor)))
        parts.append(tensor)
    return b"".join(parts)


def deserialize_parameters(data: bytes) -> Parameters:
    """Deserialize a byte buffer back to Flower ``Parameters``."""
    offset = 0
    (tt_len,) = struct.unpack_from("!I", data, offset)
    offset += 4
    tensor_type = data[offset : offset + tt_len].decode("utf-8")
    offset += tt_len
    (n_tensors,) = struct.unpack_from("!I", data, offset)
    offset += 4
    tensors: list[bytes] = []
    for _ in range(n_tensors):
        (t_len,) = struct.unpack_from("!I", data, offset)
        offset += 4
        tensors.append(data[offset : offset + t_len])
        offset += t_len
    return Parameters(tensors=tensors, tensor_type=tensor_type)


def serialize_fit_res_for_enclave(
    num_examples: int, parameters: Parameters
) -> bytes:
    """Serialize a trainer's contribution (num_examples + parameters) for the enclave."""
    return struct.pack("!I", num_examples) + serialize_parameters(parameters)


def deserialize_fit_res_for_enclave(data: bytes) -> tuple[int, Parameters]:
    """Deserialize num_examples + parameters from a byte buffer.

    Returns:
        Tuple of ``(num_examples, parameters)``.
    """
    (num_examples,) = struct.unpack_from("!I", data, 0)
    parameters = deserialize_parameters(data[4:])
    return num_examples, parameters
