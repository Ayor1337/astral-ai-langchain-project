def add(a: int, b: int) -> dict[str, int]:
    """Add two integers and return a JSON-serializable result."""
    return {"result": a + b}


def get_chat_tools() -> list:
    return [add]
