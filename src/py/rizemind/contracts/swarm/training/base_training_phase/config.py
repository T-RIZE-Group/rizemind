from pydantic import BaseModel


class BaseTrainingPhaseConfig(BaseModel):
    ttl: int

    def to_struct(self) -> dict:
        return {
            "ttl": self.ttl,
        }


class BaseEvaluationPhaseConfig(BaseModel):
    ttl: int
    registration_ttl: int

    def to_struct(self) -> dict:
        return {
            "ttl": self.ttl,
            "registrationTtl": self.registration_ttl,
        }
