"""Prompt optimization must carry the rerun's answer into rewards and output."""

import asyncio
import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tiktoken


@pytest.fixture(params=["grpo", "reinforce_plus_plus"])
def optimizer_runtime(request, monkeypatch):
    # Load the real optimizer classes while replacing service registries. Their
    # normal package imports initialize optional browser and trading backends.
    root = Path(__file__).resolve().parents[1]

    def service(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, root / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    service("src.optimizer", __path__=[str(root / "src/optimizer")])
    service("src.logger", logger=Mock())
    service("src.model", model_manager=AsyncMock())
    service(
        "src.utils", dedent=textwrap.dedent, generate_unique_id=lambda _: "test-session"
    )
    service("src.message.types", SystemMessage=Mock(), HumanMessage=Mock())
    prompt_manager = SimpleNamespace(set_variables=AsyncMock())
    service("src.prompt", prompt_manager=prompt_manager)
    service("src.tool", tcp=Mock())
    service("src.environment", ecp=Mock())
    service("src.agent", acp=Mock())
    service("src.memory", memory_manager=Mock())
    service("src.memory.types", EventType=Mock())
    session = load("src.session.types", "src/session/types.py")
    service("src.session", SessionContext=session.SessionContext)
    types = load("src.optimizer.types", "src/optimizer/types.py")
    kind = request.param
    module = load(
        f"src.optimizer.{kind}_optimizer", f"src/optimizer/{kind}_optimizer.py"
    )
    # Tokenization is not under test; avoid downloading a tokenizer.
    monkeypatch.setattr(
        tiktoken, "encoding_for_model", lambda _: SimpleNamespace(encode=list)
    )
    cls = module.GrpoOptimizer if kind == "grpo" else module.ReinforcePlusPlusOptimizer
    return kind, cls, types.Variable, prompt_manager


def agent_response(result, reasoning):
    return SimpleNamespace(
        extra=SimpleNamespace(data={"result": result, "reasoning": reasoning})
    )


@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("refine_solution", [False, True])
def test_prompt_rerun_updates_the_current_answer(
    optimizer_runtime, tmp_path, steps, refine_solution
):
    kind, cls, Variable, prompt_manager = optimizer_runtime
    reward = AsyncMock(return_value=1.0)
    optimizer = cls(
        workdir=str(tmp_path),
        memory_name=None,
        max_steps=steps,
        num_candidates=1,
        reward_fn=reward,
        optimize_solution=refine_solution,
    )
    variable = Variable(
        name="instructions",
        type="system_prompt",
        description="Instructions",
        variables="old",
    )
    optimizer.get_trainable_variables = AsyncMock(
        return_value={"instructions": variable}
    )
    optimizer._generate_reflection = AsyncMock(return_value="Update the instructions.")
    optimizer._improve_variables = AsyncMock(
        return_value={"instructions": {"variables": "new"}}
    )
    # Refinement may leave the prompt-rerun solution as it is.
    optimizer._improve_solution = AsyncMock(
        return_value=SimpleNamespace(result="", reasoning="")
    )
    optimizer._evaluate_solution = AsyncMock(
        return_value=SimpleNamespace(is_satisfied=False)
    )

    responses = [agent_response("initial", "initial reasoning")]
    expected_scored_answers = []
    for step in range(steps):
        expected_scored_answers.append("initial" if step == 0 else f"rerun {step}")
        if kind == "grpo":
            responses.append(agent_response(f"candidate {step}", "candidate reasoning"))
            expected_scored_answers.append(f"candidate {step}")
        responses.append(agent_response(f"rerun {step + 1}", f"reasoning {step + 1}"))
    agent = AsyncMock(side_effect=responses)
    agent.prompt_name = "test-prompt"
    kwargs = {"sft_solution": "reference"} if kind == "reinforce_plus_plus" else {}

    result = asyncio.run(
        optimizer.optimize(agent=agent, task="task", ground_truth="target", **kwargs)
    )

    assert result[-2:] == (f"reasoning {steps}", f"rerun {steps}")
    assert [
        call.kwargs["answer"] for call in reward.await_args_list
    ] == expected_scored_answers
    assert prompt_manager.set_variables.await_count == steps
    assert optimizer._evaluate_solution.await_count == steps
    for step, call in enumerate(optimizer._evaluate_solution.await_args_list, start=1):
        assert (
            call.kwargs["execution_result"]
            == f"Result: rerun {step}\nReasoning: reasoning {step}"
        )
