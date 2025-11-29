from datasets import Dataset
from sentence_transformers.sparse_encoder import (
    SparseEncoder,
    SparseEncoderTrainer,
    losses,
)
from sentence_transformers.sparse_encoder.models import SparseAutoEncoder
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from torch import Tensor
import torch


class SAEWrapper(SentenceTransformer):
    """
    Inference wrapper that returns sparse latents.

    teacher_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    teacher_model_dim = teacher_model.encode("text").shape[0]

    sae_wrapper = SAEWrapper(
        teacher_model,
        SparseAutoEncoder(input_dim=teacher_model_dim, hidden_dim=teacher_model_dim*16, k=32)
    )
    """

    def __init__(
        self, sentence_transformer: SentenceTransformer, sae: SparseAutoEncoder
    ):
        super().__init__(modules=[*sentence_transformer, sae])

    @property
    def sae(self) -> SparseAutoEncoder:
        return list(self._modules.values())[-1]

    @property
    def teacher(self) -> SentenceTransformer:
        """Get the teacher model (all modules except SAE)"""
        return SentenceTransformer(modules=list(self._modules.values())[:-1])

    def forward(
        self, features: dict[str, Tensor], max_active_dims: int | None = None
    ) -> dict[str, Tensor]:
        modules = list(self._modules.values())
        for module in modules[:-1]:
            features = module(features)
        sae_output = self.sae(features, max_active_dims=max_active_dims)
        features["sentence_embedding"] = sae_output["sentence_embedding"]
        return features

    def decode(self, latents: Tensor) -> Tensor:
        return self.sae.decode(latents)


class EmbeddingReconstructionLoss(nn.Module):
    def __init__(self, teacher: SentenceTransformer, autoencoder: SparseAutoEncoder):
        super().__init__()
        self.teacher = teacher
        self.autoencoder = autoencoder
        self.loss_fct = nn.MSELoss()

        # Freeze teacher - only train autoencoder
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, sentence_features: list[dict[str, Tensor]], labels=None) -> Tensor:
        features = sentence_features[0]

        # 1. Get teacher embeddings (dense target)
        with torch.no_grad():
            teacher_out = self.teacher(features)
            target = teacher_out["sentence_embedding"]

        # 2. SAE: dense → sparse → reconstruction
        sae_out = self.autoencoder({"sentence_embedding": target})
        sparse_latents = sae_out["sentence_embedding"]
        reconstruction = self.autoencoder.decode(sparse_latents)

        # 3. MSE between reconstruction and target
        return self.loss_fct(reconstruction, target)
