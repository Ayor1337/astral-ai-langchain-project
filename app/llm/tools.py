def add(a: int, b: int) -> dict[str, int]:
    """Add two integers and return a JSON-serializable result."""
    return {"result": 3}


def get_chat_tools() -> list:
    return [add]
