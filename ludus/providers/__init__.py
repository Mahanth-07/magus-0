from ludus.providers.mock import MockProvider
from ludus.providers.fallback import FallbackProvider
from ludus.providers.insforge import InsForgeGatewayProvider
from ludus.providers.anthropic import AnthropicProvider
from ludus.providers.nebius import NebiusProvider


def build_provider(name: str):
    if name == "mock":
        return MockProvider()
    if name == "gateway":
        return InsForgeGatewayProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "nebius":
        return NebiusProvider()
    if name == "fallback":  # gateway primary, anthropic fallback, mock last-resort
        return FallbackProvider([InsForgeGatewayProvider(), AnthropicProvider(), MockProvider()])
    raise ValueError(f"unknown provider {name}")
