"""Download-free tests for pooling scalar token logits from layerwise models."""

import pytest
import torch
from transformers import BertTokenizer, GPT2Config, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithPast

from rerankers.documents import Document
from rerankers.models.llm_layerwise_ranker import LLMLayerWiseRanker


class TinyLayerwiseModel(torch.nn.Module):
    """Real tiny decoder with scalar heads exposing the layerwise output contract."""

    def __init__(self, vocab_size):
        super().__init__()
        self.decoder = GPT2Model(
            GPT2Config(
                vocab_size=vocab_size,
                n_positions=128,
                n_embd=16,
                n_layer=2,
                n_head=2,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                use_cache=False,
            )
        )
        self.head = torch.nn.Linear(16, 1, bias=False)

    def forward(self, input_ids, attention_mask, cutoff_layers, return_dict=True):
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = tuple(
            self.head(outputs.hidden_states[layer]).squeeze(-1)
            for layer in cutoff_layers
        )
        return CausalLMOutputWithPast(logits=logits)


@pytest.fixture
def ranker(tmp_path):
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "a",
        "b",
        ":",
        "query",
        "short",
        "long",
        "document",
        "relevant",
        "yes",
        "no",
    ]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizer(str(vocab_path), bos_token="[CLS]")
    with torch.random.fork_rng():
        torch.manual_seed(123)
        model = TinyLayerwiseModel(len(tokenizer)).eval()
    ranker = object.__new__(LLMLayerWiseRanker)
    ranker.model = model
    ranker.tokenizer = tokenizer
    ranker.device = "cpu"
    ranker.dtype = torch.float32
    ranker.batch_size = 2
    ranker.max_sequence_length = 64
    ranker.prompt = "relevant"
    ranker.params = {"cutoff_layers": [1, 2]}
    return ranker


@torch.inference_mode()
def unpadded_score(ranker, query, doc):
    """Independent oracle: physically remove padding before the forward pass."""
    inputs = ranker._get_inputs([(query, doc)], ranker.max_sequence_length)
    mask = inputs["attention_mask"][0].bool()
    input_ids = inputs["input_ids"][:, mask]
    outputs = ranker.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        return_dict=True,
        **ranker.params,
    )
    return outputs.logits[-1][0, -1].item()


@pytest.mark.parametrize("doc", ["", "short", "long document", "document " * 70])
@pytest.mark.parametrize("cutoff_layers", [[1], [2], [1, 2]])
def test_score_uses_final_prompt_token(ranker, doc, cutoff_layers):
    ranker.params["cutoff_layers"] = cutoff_layers
    assert ranker.score("query", doc) == pytest.approx(
        unpadded_score(ranker, "query", doc), abs=1e-6
    )


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
@pytest.mark.parametrize("cutoff_layers", [[1], [2], [1, 2]])
def test_rank_matches_unpadded_scores(ranker, batch_size, cutoff_layers):
    ranker.params["cutoff_layers"] = cutoff_layers
    texts = [
        "short",
        "long document document document document document",
        "",
        "document " * 70,
    ]
    docs = [
        Document(text=text, doc_id=f"doc-{i}", metadata={"index": i})
        for i, text in enumerate(texts)
    ]
    expected = {doc.doc_id: unpadded_score(ranker, "query", doc.text) for doc in docs}
    results = ranker.rank("query", docs, batch_size=batch_size)
    assert len(results.results) == len(docs)
    assert [result.rank for result in results.results] == [1, 2, 3, 4]
    assert [result.score for result in results.results] == sorted(
        [result.score for result in results.results], reverse=True
    )
    for result in results.results:
        assert result.score == pytest.approx(expected[result.doc_id], abs=1e-6)
        assert result.document is docs[result.metadata["index"]]


def test_singleton_multiple_of_eight_padding(ranker):
    inputs = ranker._get_inputs([("query", "long document")], 64)
    assert inputs["input_ids"].shape[1] % 8 == 0
    assert inputs["attention_mask"][0, -1].item() == 0
    expected = unpadded_score(ranker, "query", "long document")
    assert ranker.score("query", "long document") == pytest.approx(expected, abs=1e-6)


def test_compressed_output_uses_its_own_final_layer_mask(ranker):
    # Compression changes sequence lengths, so input-token indices are invalid.
    inputs = {
        "input_ids": torch.ones((2, 16), dtype=torch.long),
        "attention_mask": torch.tensor([[1] * 10 + [0] * 6, [1] * 13 + [0] * 3]),
    }
    outputs = CausalLMOutputWithPast(
        logits=(
            torch.full((2, 8), -10.0),
            torch.tensor([[1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 88.0, 77.0]]),
        )
    )
    outputs.attention_masks = (
        torch.ones((2, 8), dtype=torch.long),
        torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )
    ranker.model = lambda **kwargs: outputs
    ranker._get_inputs = lambda *args, **kwargs: inputs
    results = ranker.rank("query", ["short", "long"])
    assert [(result.doc_id, result.score) for result in results.results] == [
        (1, 5.0),
        (0, 3.0),
    ]


def test_constructor_forces_right_padding(ranker, monkeypatch):
    ranker.tokenizer.padding_side = "left"
    monkeypatch.setattr(
        "rerankers.models.llm_layerwise_ranker.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: ranker.tokenizer,
    )
    monkeypatch.setattr(
        "rerankers.models.llm_layerwise_ranker.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: ranker.model,
    )
    initialized = LLMLayerWiseRanker("local-tiny", device="cpu", verbose=0)
    assert initialized.tokenizer.padding_side == "right"
