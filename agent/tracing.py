import os
import logfire

def setup_tracing() -> None:
    """
    Configure logfire once for entire app
    """
    logfire.configure(
        token = os.environ.get("LOGFIRE_TOKEN"),
        service_name = "multi-agent-system"
    )
    logfire.instrument_openai()