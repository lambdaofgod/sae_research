import os

os.environ["WANDB_DISABLED"] = "true"

import pytest
import torch
from datasets import Dataset, load_dataset
from datasets import Dataset
from sentence_transformers.sparse_encoder import SparseEncoderTrainer
from sentence_transformers.sparse_encoder.models import SparseAutoEncoder
from sentence_transformers import SentenceTransformer
from sae_research.embedder_sae import SAEWrapper, EmbeddingReconstructionLoss
from sentence_transformers.sparse_encoder import (
    SparseEncoder,
    SparseEncoderTrainer,
    losses,
    SparseEncoderTrainingArguments,
)


@pytest.mark.slow
def test_sae_setup():
    teacher_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    teacher_model_reference = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    teacher_model_dim = teacher_model.encode("text").shape[0]

    hidden_dim = teacher_model_dim * 16
    sae = SparseAutoEncoder(input_dim=teacher_model_dim, hidden_dim=hidden_dim, k=32)

    # SAEWrapper for inference (returns sparse)
    sae_wrapper = SAEWrapper(teacher_model, sae)
    assert sae_wrapper.encode(["This is a test sentence."]).shape[1] == hidden_dim

    train_dataset = load_dataset("HuggingFaceFW/fineweb", streaming=True)["train"].select_columns(["text"])
    # Loss uses teacher + SAE module (not wrapper)
    loss = EmbeddingReconstructionLoss(teacher=teacher_model, autoencoder=sae)
    args = SparseEncoderTrainingArguments(
        max_steps=2,  # Train for 20k steps total
    )

    trainer = SparseEncoderTrainer(
        model=sae_wrapper, train_dataset=train_dataset, loss=loss, args=args
    )
    trainer.train()

    # Verify teacher model wasn't modified during training
    for p1, p2 in zip(sae_wrapper.teacher.parameters(), teacher_model_reference.parameters()):
        assert torch.equal(p1, p2), "Teacher model was modified during training!"
