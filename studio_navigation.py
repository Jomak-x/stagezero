"""Small navigation adapter for the pinned Viser client protocol."""
import uuid
from viser._messages import GuiUpdateMessage


def navigate_tab(tabs, index, client):
    """Navigate only the client that clicked; don't move other viewers' tabs."""
    connection = getattr(client, '_websock_connection', None)
    state = getattr(tabs, '_impl', None)
    if connection is None or state is None:
        return
    connection.queue_message(GuiUpdateMessage(state.uuid, {
        '_studio_tab_request': {'index': int(index), 'nonce': uuid.uuid4().hex},
    }))
