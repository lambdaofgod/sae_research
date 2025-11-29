import json
import os

import fire
from pydantic import BaseModel
from datasets import load_dataset, IterableDataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sparse_encoder import (
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.models import SparseAutoEncoder

from sae_research.embedder_sae import (
    SAEWrapper,
    EmbeddingReconstructionLoss,
    NormalizedEmbeddingReconstructionLoss,
)


class SAEEmbedderTrainingConfig(BaseModel):
    teacher_model_name: str
    hidden_dim: int | None
    k: int
    dataset_name: str
    text_column: str
    max_steps: int
    batch_size: int
    output_dir: str | None
    disable_wandb: bool
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    use_normalized_loss: bool


def create_sae_wrapper(model_name: str, hidden_dim: int | None, k: int) -> SAEWrapper:
    teacher = SentenceTransformer(model_name)
    embedding_dim = teacher.encode("text").shape[0]
    actual_hidden_dim = hidden_dim if hidden_dim is not None else 8 * embedding_dim
    sae = SparseAutoEncoder(input_dim=embedding_dim, hidden_dim=actual_hidden_dim, k=k)
    return SAEWrapper(teacher, sae)


def load_dataset_streaming(dataset_name: str, text_column: str) -> IterableDataset:
    return (
        load_dataset(dataset_name, streaming=True)["train"]
        .select_columns([text_column])
        .with_format("torch")
    )


def get_output_dir(output_dir: str | None, model_name: str) -> str:
    generated_name = "sae_" + model_name.replace("/", "_")
    if output_dir is not None:
        return os.path.join(output_dir, generated_name)
    return generated_name


def train(config: SAEEmbedderTrainingConfig):
    if config.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    sae_wrapper = create_sae_wrapper(
        config.teacher_model_name, config.hidden_dim, config.k
    )
    dataset = load_dataset_streaming(config.dataset_name, config.text_column)

    loss_cls = (
        NormalizedEmbeddingReconstructionLoss
        if config.use_normalized_loss
        else EmbeddingReconstructionLoss
    )
    loss = loss_cls(teacher=sae_wrapper.teacher, autoencoder=sae_wrapper.sae)
    args = SparseEncoderTrainingArguments(
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=True,
    )
    trainer = SparseEncoderTrainer(
        model=sae_wrapper, train_dataset=dataset, loss=loss, args=args
    )
    trainer.train()

    output_dir = get_output_dir(config.output_dir, config.teacher_model_name)
    sae_wrapper.save(output_dir)

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)


def main(
    teacher_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    hidden_dim: int | None = None,
    k: int = 16,
    dataset_name: str = "HuggingFaceFW/fineweb",
    text_column: str = "text",
    max_steps: int = 20000,
    batch_size: int = 1024,
    output_dir: str | None = None,
    disable_wandb: bool = True,
    dataloader_num_workers: int = 8,
    dataloader_prefetch_factor: int = 4,
    use_normalized_loss: bool = True,
):
    config = SAEEmbedderTrainingConfig(
        teacher_model_name=teacher_model_name,
        hidden_dim=hidden_dim,
        k=k,
        dataset_name=dataset_name,
        text_column=text_column,
        max_steps=max_steps,
        batch_size=batch_size,
        output_dir=output_dir,
        disable_wandb=disable_wandb,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        use_normalized_loss=use_normalized_loss,
    )
    train(config)


if __name__ == "__main__":
    fire.Fire(main)
