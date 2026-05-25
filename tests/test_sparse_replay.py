# SPDX-License-Identifier: MIT

from atom.utils.debug_helper import sparse_replay


def test_should_log_layer_skips_during_cuda_graph_capture(monkeypatch):
    monkeypatch.setenv("ATOM_SPARSE_REPLAY_C_PATH", "/tmp/sparse_replay.jsonl")
    monkeypatch.setattr(
        sparse_replay.torch.cuda, "is_current_stream_capturing", lambda: True
    )

    assert sparse_replay.should_log_layer(0) is False


def test_should_log_layer_ignores_unavailable_capture_query(monkeypatch):
    monkeypatch.setenv("ATOM_SPARSE_REPLAY_C_PATH", "/tmp/sparse_replay.jsonl")

    def raise_runtime_error():
        raise RuntimeError("CUDA driver is unavailable")

    monkeypatch.setattr(
        sparse_replay.torch.cuda, "is_current_stream_capturing", raise_runtime_error
    )

    assert sparse_replay.should_log_layer(0) is True


def test_maybe_log_skips_during_cuda_graph_capture(monkeypatch, tmp_path):
    log_path = tmp_path / "sparse_replay.jsonl"
    monkeypatch.setenv("ATOM_SPARSE_REPLAY_C_PATH", str(log_path))
    monkeypatch.setattr(
        sparse_replay.torch.cuda, "is_current_stream_capturing", lambda: True
    )

    sparse_replay.maybe_log("capture_event", value=1)

    assert not log_path.exists()
