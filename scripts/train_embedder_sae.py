import fire
from pydantic import BaseModel
from datasets import load_dataset, IterableDataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sparse_encoder import SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder.models import SparseAutoEncoder

from sae_research.embedder_sae import SAEWrapper, EmbeddingReconstructionLoss


class SAEEmbedderTrainingConfig(BaseModel):
    teacher_model_name: str
    hidden_dim: int | None
    k: int
    dataset_name: str
    text_column: str
    max_steps: int
    output_dir: str | None


def create_sae_wrapper(model_name: str, hidden_dim: int | None, k: int) -> SAEWrapper:
    teacher = SentenceTransformer(model_name)
    embedding_dim = teacher.encode("text").shape[0]
    actual_hidden_dim = hidden_dim if hidden_dim is not None else 16 * embedding_dim
    sae = SparseAutoEncoder(input_dim=embedding_dim, hidden_dim=actual_hidden_dim, k=k)
    return SAEWrapper(teacher, sae)


def load_dataset_streaming(dataset_name: str, text_column: str) -> IterableDataset:
    return load_dataset(dataset_name, streaming=True)["train"].select_columns([text_column])


def get_output_dir(output_dir: str | None, model_name: str) -> str:
    if output_dir is not None:
        return output_dir
    return "sae_" + model_name.replace("/", "_")


def train(config: SAEEmbedderTrainingConfig):
    sae_wrapper = create_sae_wrapper(config.teacher_model_name, config.hidden_dim, config.k)
    dataset = load_dataset_streaming(config.dataset_name, config.text_column)

    loss = EmbeddingReconstructionLoss(teacher=sae_wrapper.teacher, autoencoder=sae_wrapper.sae)
    args = SparseEncoderTrainingArguments(max_steps=config.max_steps)
    trainer = SparseEncoderTrainer(model=sae_wrapper, train_dataset=dataset, loss=loss, args=args)
    trainer.train()

    output_dir = get_output_dir(config.output_dir, config.teacher_model_name)
    sae_wrapper.save(output_dir)


def main(
    teacher_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    hidden_dim: int | None = None,
    k: int = 32,
    dataset_name: str = "HuggingFaceFW/fineweb",
    text_column: str = "text",
    max_steps: int = 20000,
    output_dir: str | None = None,
):
    config = SAEEmbedderTrainingConfig(
        teacher_model_name=teacher_model_name,
        hidden_dim=hidden_dim,
        k=k,
        dataset_name=dataset_name,
        text_column=text_column,
        max_steps=max_steps,
        output_dir=output_dir,
    )
    train(config)


if __name__ == "__main__":
    fire.Fire(main)
