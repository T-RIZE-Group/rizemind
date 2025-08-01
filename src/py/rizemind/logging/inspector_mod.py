from flwr.client.typing import ClientAppCallable
from flwr.common import (
    Context,
    Message,
)


def inspector_mod(
    msg: Message,
    ctxt: Context,
    call_next: ClientAppCallable,
) -> Message:
    print("new message")
    print(msg.metadata.message_type)
    print(msg.content.keys())
    response = call_next(msg, ctxt)
    print("response")
    print(response.metadata.message_type)
    print(response.content.keys())
    return response
