from .loop import InferenceLoop
from ..pipeline import Pipeline


class BSRInferenceLoop(InferenceLoop):
    """Upsample-only + ControlNet diffusion for super-resolution.
    No Stage-1 reconstruction model (SwinIR/BSRNet) is used.
    """

    def load_cleaner(self) -> None:
        self.cleaner = None

    def load_pipeline(self) -> None:
        self.pipeline = Pipeline(
            self.cleaner,
            self.cldm,
            self.diffusion,
            self.cond_fn,
            self.args.device,
            self.args.upscale,
        )
