from ludus.providers.mock import MockProvider
from ludus.providers.fallback import FallbackProvider
from ludus.providers.insforge import InsForgeGatewayProvider
from ludus.providers.anthropic import AnthropicProvider


def build_provider(name: str):
    if name == "mock":
        return MockProvider()
    if name == "gateway":
        return InsForgeGatewayProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "fallback":  # gateway primary, anthropic fallback, mock last-resort
        return FallbackProvider([InsForgeGatewayProvider(), AnthropicProvider(), MockProvider()])
    raise ValueError(f"unknown provider {name}")
