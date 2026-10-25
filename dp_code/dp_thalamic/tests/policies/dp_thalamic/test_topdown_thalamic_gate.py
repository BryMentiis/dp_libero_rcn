#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch

from lerobot.policies.dp_thalamic.topdown_thalamic_gate import TopDownThalamicGate1d


@pytest.mark.parametrize("channels,length", [(2048, 8), (1024, 16)])
def test_gate_shape(channels, length):
    gate = TopDownThalamicGate1d(channels, max(16, channels // 16))
    skip = torch.randn(2, channels, length)
    context = torch.randn(2, channels, length)
    out = gate(skip, context)
    assert out.shape == skip.shape
    assert torch.isfinite(out).all()


def test_identity_at_initialization():
    gate = TopDownThalamicGate1d(64, 16)
    skip = torch.randn(3, 64, 8)
    context = torch.randn(3, 64, 8)
    out = gate(skip, context)
    torch.testing.assert_close(out, skip, rtol=0, atol=0)


def test_active_gate_and_gradients():
    gate = TopDownThalamicGate1d(64, 16)
    with torch.no_grad():
        gate.gamma.fill_(1.0)
    skip = torch.randn(2, 64, 8, requires_grad=True)
    context = torch.randn(2, 64, 8, requires_grad=True)
    out = gate(skip, context)
    assert not torch.allclose(out, skip)
    out.square().mean().backward()
    assert gate.gamma.grad is not None
    assert skip.grad is not None
    assert context.grad is not None
    assert all(p.grad is not None for p in gate.gate_logits.parameters())


def test_invalid_inputs_raise():
    gate = TopDownThalamicGate1d(64, 16)
    with pytest.raises(ValueError):
        gate(torch.randn(2, 64), torch.randn(2, 64))
    with pytest.raises(ValueError):
        gate(torch.randn(2, 64, 8), torch.randn(2, 64, 4))
    with pytest.raises(ValueError):
        gate(torch.randn(2, 32, 8), torch.randn(2, 32, 8))
