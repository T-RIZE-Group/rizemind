from hexbytes import HexBytes
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


def _hexbytes_schema(cls, source, handler: GetCoreSchemaHandler):
    def _validate(v):
        if isinstance(v, HexBytes):
            return v

        raise TypeError(f"Cannot parse {type(v)} as HexBytes")

    # validate to HexBytes in Python; serialize to a hex string in JSON
    return core_schema.no_info_plain_validator_function(
        _validate,
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda v: v,
            return_schema=core_schema.bytes_schema(),
            when_used="json",
        ),
    )


# apply once at import time
HexBytes.__get_pydantic_core_schema__ = classmethod(_hexbytes_schema)  # pyright: ignore[reportAttributeAccessIssue]
