"""Download-free tests for the CostWiseGemma input segmentation contract."""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rerankers.models.llm_layerwise_ranker import LLMLayerWiseRanker


@pytest.fixture(params=[False, True], ids=["newline", "multi-token-newline"])
def tokenizer(request):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokens = ["[PAD]", "[UNK]", "[BOS]"] + alphabet
    backend = Tokenizer(
        models.BPE({token: i for i, token in enumerate(tokens)}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if request.param:
        backend.normalizer = normalizers.Replace("\n", "<NL>")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        model_max_length=512,
    )


def make_ranker(tokenizer, model_type="cost_wise_gemma", prompt="relevant"):
    ranker = object.__new__(LLMLayerWiseRanker)
    ranker.tokenizer = tokenizer
    ranker.prompt = prompt
    # Only configuration metadata is needed for these input-construction tests.
    ranker.model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    return ranker


def expected_segments(tokenizer, query, passage, prompt, max_length):
    def encode(text, max_length=None):
        return tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=max_length is not None,
            max_length=max_length,
        )

    query_ids = [tokenizer.bos_token_id] + encode(f"A: {query}", max_length * 3 // 4)
    separator = encode("\n")
    passage_ids = encode(f"B: {passage}", max_length)
    second = separator + passage_ids
    keep = max_length - len(query_ids)
    removed = max(0, len(second) - keep)
    if tokenizer.truncation_side == "left":
        second = second[removed:]
        separator_count = max(0, len(separator) - removed)
    else:
        second = second[:keep]
        separator_count = min(len(separator), len(second))
    return (
        query_ids + second[:separator_count],
        second[separator_count:],
        separator + encode(prompt),
    )


@pytest.mark.parametrize("truncation_side", ["left", "right"])
@pytest.mark.parametrize("max_length", [12, 32, 64])
@pytest.mark.parametrize("prompt", ["", "answer yes or no", "関連性"])
def test_preserves_query_and_prompt_boundaries(
    tokenizer, truncation_side, max_length, prompt
):
    tokenizer.truncation_side = truncation_side
    ranker = make_ranker(tokenizer, prompt=prompt)
    pairs = [
        ("", ""),
        ("query", "short"),
        ("long query " * 20, "long passage " * 30),
        ("query", "long passage " * 30),
    ]
    inputs = ranker._get_inputs(pairs, max_length)
    assert inputs["query_lengths"].dtype == torch.long
    assert inputs["prompt_lengths"].dtype == torch.long
    assert inputs["query_lengths"].shape == (len(pairs),)
    assert inputs["prompt_lengths"].shape == (len(pairs),)
    assert inputs["input_ids"].shape[1] % 8 == 0
    for row, (query, passage) in enumerate(pairs):
        expected_query, expected_passage, expected_prompt = expected_segments(
            tokenizer, query, passage, prompt, max_length
        )
        ids = inputs["input_ids"][row][inputs["attention_mask"][row].bool()].tolist()
        q = inputs["query_lengths"][row].item()
        p = inputs["prompt_lengths"][row].item()
        assert q == len(expected_query)
        assert p == len(expected_prompt)
        assert ids == expected_query + expected_passage + expected_prompt
        assert ids[:q] == expected_query
        assert ids[q : len(ids) - p] == expected_passage
        assert ids[len(ids) - p :] == expected_prompt


@pytest.mark.parametrize("params", [{}, {"compress_ratio": 1}, {"compress_layer": []}])
def test_costwise_metadata_does_not_depend_on_requested_compression(tokenizer, params):
    ranker = make_ranker(tokenizer)
    ranker.params = params
    # The configuration identifies local and renamed checkpoints as well.
    ranker.model.config._name_or_path = "/local/renamed-checkpoint"
    inputs = ranker._get_inputs([("query", "passage")], 64)
    assert set(inputs) == {
        "input_ids",
        "attention_mask",
        "query_lengths",
        "prompt_lengths",
    }


@pytest.mark.parametrize("model_type", ["minicpm", "gemma", "gemma2", "gpt2", None])
def test_other_models_do_not_receive_compression_metadata(tokenizer, model_type):
    ranker = make_ranker(tokenizer, model_type=model_type)
    inputs = ranker._get_inputs([("query", "passage")], 64)
    assert set(inputs) == {"input_ids", "attention_mask"}


def test_model_without_config_keeps_existing_inputs(tokenizer):
    ranker = make_ranker(tokenizer)
    ranker.model = torch.nn.Identity()
    inputs = ranker._get_inputs([("query", "passage")], 64)
    assert set(inputs) == {"input_ids", "attention_mask"}
