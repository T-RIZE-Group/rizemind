from attr import dataclass
from rizemind.swarm.indexer.base_swarm_event import SwarmEvent


@dataclass
class NewPhaseEvent(SwarmEvent):
    phase: str
