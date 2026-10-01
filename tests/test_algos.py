"""Learning code: masking, mixer monotonicity, attention, smoke training, pipeline."""
import json
import os

import numpy as np
import pytest
import torch

from sim.blackstart_env import BlackStartEnv
from sim.config import load
from sim.nets import DAN, QMixer
from sim.train import SPECS, dims_of, make_agent, make_env, run


def _cfg():
    c = load()
    c["train"].update(probe_every=4, probe_episodes=2, eval_episodes=3, batch_episodes=2)
    return c


@pytest.mark.parametrize("algo", ["pi-mappo", "mappo", "ppo", "dqn", "iql", "vdn", "qmix"])
def test_smoke_training(algo):
    r = run(algo, "14", 0, episodes=4, cfg=_cfg(), scenarios=False)
    assert len(r["hist"]["ret"]) == 4 and np.isfinite(r["test"]["ret"])


@pytest.mark.parametrize("algo", ["pi-mappo", "qmix+pi"])
def test_masked_agents_never_select_invalid_actions(algo):
    cfg = _cfg()
    env = make_env(algo, "14", cfg)
    agent = make_agent(algo, env, cfg, 0)
    obs, state = env.reset(1)
    for _ in range(10):
        m = env.action_masks()
        a = agent.act(obs, state, m, greedy=True)
        a = a[0] if isinstance(a, tuple) else a
        assert all(m[i][a[i]] for i in range(5))
        obs, state, _, done, _ = env.step(a)
        if done:
            break


def test_qmix_is_monotone():
    mix = QMixer(5, 12)
    q = torch.randn(16, 5, requires_grad=True)
    mix(q, torch.randn(16, 12)).sum().backward()
    assert (q.grad >= -1e-7).all()


def test_dan_attention_is_a_distribution_over_the_other_agents():
    dan = DAN([7, 4, 4, 5, 9])
    out = dan([torch.randn(3, d) for d in (7, 4, 4, 5, 9)])
    w = dan.last_attention
    assert all(o.shape == (3, dan.out_dim) for o in out)
    assert torch.allclose(w.sum(-1), torch.ones(3, 5), atol=1e-6) and float(w.diagonal(dim1=1, dim2=2).abs().max()) < 1e-6


def test_every_spec_declares_its_components():
    for name, s in SPECS.items():
        assert set(s) == {"fam", "kind", "mono", "mask", "feat", "shape", "phys", "dan"}, name
    assert all(SPECS["pi-mappo"][k] for k in ("mask", "feat", "shape", "phys", "dan"))
    assert not any(SPECS["mappo"][k] for k in ("mask", "feat", "shape", "phys", "dan"))


def test_pipeline_end_to_end(tmp_path):
    from sim import make_tables, run_experiments as rx
    cfg = _cfg()
    rx.run_baselines(["14"], cfg, str(tmp_path), n_beam=0)
    rx._matrix("benchmark", ["vdn", "pi-mappo"], ["14"], [0, 1], 4, cfg, 1, str(tmp_path), stress_system="14")
    rx._dump(dict(label="TEST", seeds=[0, 1], episodes=4, commit="x"), str(tmp_path), "meta.json")
    import sys
    argv, sys.argv = sys.argv, ["x", "--results", str(tmp_path)]
    try:
        make_tables.main()
    finally:
        sys.argv = argv
    b = json.load(open(tmp_path / "benchmark.json"))
    assert set(b["14"]["pi-mappo"]["scenarios"]) == {"S1", "S2", "S3", "S4", "S5"}
    assert "ret_shaped" in b["14"]["pi-mappo"]["hist"] and "stress" in b["14"]["pi-mappo"]
    for fn in ("table_env.tex", "table_multisystem.tex", "table_improvement.tex", "table_baselines.tex", "claims.md"):
        assert os.path.getsize(tmp_path / fn) > 50
