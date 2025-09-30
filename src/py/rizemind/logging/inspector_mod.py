from flwr.client.typing import ClientAppCallable
from flwr.common import (
    Context,
    Message,
)


def inspector_mod(
    msg: Message,
    ctx: Context,
    call_next: ClientAppCallable,
) -> Message:
    """Prints out the incoming message and outgoing response.

    If you have more than one mode, the order of the `inspector_mod`
    must be considered against them, the output would be different
    if the order is changed.

    Args:
        msg: The incoming message from the ServerApp to the ClientApp.
        ctx: Context of the run.
        call_next: The function that gets executed next to generate
            the response to the incoming message and context.

    Returns:
        The response message sent from the ClientApp to the ServerApp.
    """
    print("new message")
    print(msg.metadata.message_type)
    print(msg.content.keys())
    response = call_next(msg, ctx)
    print("response")
    print(response.metadata.message_type)
    print(response.content.keys())
    return response
