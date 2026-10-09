import json

import torch
from tokenizers import Tokenizer
from transformers import Qwen2Config, Qwen2ForCausalLM

from lipi import evaluate, extend, model as M
from tests.test_extend import QWEN, tok


def tiny(vocab):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).eval()


def extended_pair():
    base = extend.with_marks(tok(QWEN))
    words = extend.words(Tokenizer.from_str(json.dumps(base)), ["ମଧୁମେହ ଆକ୍ରାନ୍ତ"] * 4, extend.odia)
    merges = extend.confine(extend.learn(base, words, 30), base["model"]["vocab"], extend.odia_bytes)
    ext, added = extend.apply(base, merges)
    return base, ext, added


def test_new_rows_start_at_the_mean_of_their_pieces():
    base, ext, added = extended_pair()
    m = tiny(len(base["model"]["vocab"]))
    old = m.get_input_embeddings().weight.detach().clone()
    sources = M.extend_embeddings(m, base, ext, multiple=8)
    emb = m.get_input_embeddings().weight
    assert emb.shape[0] % 8 == 0 and emb.shape[0] >= len(ext["model"]["vocab"])
    assert torch.equal(emb[:old.shape[0]], old)                              # old rows untouched
    assert m.get_output_embeddings().weight.data_ptr() == emb.data_ptr()     # still tied
    t = Tokenizer.from_str(json.dumps(ext))
    i = t.token_to_id(added[-1])
    assert torch.allclose(emb[i], old[sources[i]].mean(0))
    assert len(sources[i]) > 1


def test_continuation_logprobs_match_full_logits():
    m = tiny(300)
    prefixes, conts = [[1, 2, 3], [4, 5]], [[6, 7], [8, 9, 10, 11]]
    got = evaluate.continuation_logprobs(m, prefixes, conts, pad=0, batch=2)
    for p, c, g in zip(prefixes, conts, got):
        logp = torch.log_softmax(m(input_ids=torch.tensor([p + c])).logits[0].float(), -1)
        want = sum(logp[len(p) - 1 + k, c[k]].item() for k in range(len(c)))
        assert abs(g - want) < 1e-4


def test_chunked_training_loss_matches_the_plain_loss_and_gradients():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("train_m1", os.path.join(os.path.dirname(__file__), "..", "scripts", "train_m1.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    batch = torch.randint(0, 500, (2, 40))
    ref = tiny(500)
    logits = ref(input_ids=batch[:, :-1]).logits
    plain = torch.nn.functional.cross_entropy(logits.reshape(-1, 500), batch[:, 1:].reshape(-1))
    plain.backward()
    g = ref.get_input_embeddings().weight.grad
    for n_base in (500, 420):                     # no new rows; 80 new rows held apart
        m = tiny(500)
        lm = train.LossModel(m, n_base, chunk=7)
        loss = lm(batch)
        loss.backward()
        assert abs(loss.item() - plain.item()) < 1e-5
        assert torch.allclose(m.get_input_embeddings().weight.grad, g[:n_base], atol=1e-6)
        if n_base < 500:
            assert torch.allclose(lm.new.grad, g[n_base:], atol=1e-6)
        lm.merge()
        assert m.get_input_embeddings().weight.shape[0] == 500
        assert m.get_output_embeddings().weight is m.get_input_embeddings().weight
        assert torch.allclose(m(input_ids=batch[:, :-1]).logits, logits, atol=1e-5)
