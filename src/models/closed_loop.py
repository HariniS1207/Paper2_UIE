import torch
import torch.nn as nn

from src.models.controllable_enhancer import ControllableEnhancer
from src.models.redegradation import ReDegradation
from src.models.consequence_encoder import ConsequenceEncoder
from src.models.feedback_controller import FeedbackController


class OneStepClosedLoop(nn.Module):
    """
    One-step consequence-aware enhancement loop.

    X -> E(X, alpha) -> Y -> R(Y)
      -> consequence -> feedback -> alpha'
    """

    def __init__(self):
        super().__init__()

        self.enhancer = ControllableEnhancer()
        self.redegrader = ReDegradation()
        self.consequence_encoder = ConsequenceEncoder()
        self.feedback_controller = FeedbackController()

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
    ):
        enhanced = self.enhancer(x, alpha)

        redegraded = self.redegrader(enhanced)

        consequence_map, consequence_vector = (
            self.consequence_encoder(
                x,
                redegraded,
            )
        )

        updated_alpha = self.feedback_controller(
            consequence_vector,
            alpha,
        )

        return {
            "enhanced": enhanced,
            "redegraded": redegraded,
            "consequence_map": consequence_map,
            "consequence_vector": consequence_vector,
            "alpha": alpha,
            "updated_alpha": updated_alpha,
        }