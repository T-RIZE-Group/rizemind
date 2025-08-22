from rizemind.contracts.sampling.selector_factory import SelectorConfig

ALWAYS_SAMPLED_SELECTOR = "always-sampled"


class AlwaysSamplesSelectorConfig(SelectorConfig):
    name: str = ALWAYS_SAMPLED_SELECTOR
    version: str = "1.0.0"
