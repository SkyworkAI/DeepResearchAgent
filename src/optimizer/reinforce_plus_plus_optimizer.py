import math
import tiktoken
import json
import os
import uuid
from datetime import datetime
from typing import List, Optional, Any, Dict, Union, Callable, TYPE_CHECKING
from pydantic import ConfigDict, Field
from pydantic import BaseModel

from src.logger import logger
from src.optimizer.types import Optimizer, Variable
from src.model import model_manager
from src.message.types import SystemMessage, HumanMessage
from src.utils import dedent

if TYPE_CHECKING:
    from src.session import SessionContext


class Response(BaseModel):
    reasoning: str = Field(description="The reasoning process")
    result: str = Field(description="The final result")

class ImprovedVariable(BaseModel):
    name: str = Field(description="The name of the variable")
    variables: str = Field(description="The value of the variable")

class ImprovedVariables(BaseModel):
    variables: List[ImprovedVariable] = Field(default=[], description="The variables to improve")

class EvaluationResult(BaseModel):
    is_satisfied: bool = Field(description="Whether the current solution and variables are satisfactory and optimization can stop")
    reason: str = Field(description="The reason for the decision")

class ReinforcePlusPlusOptimizer(Optimizer):
    """Optimizer that improves agent prompts using the REINFORCE++ method."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    prompt_name: str = Field(default="reflection_optimizer", description="The name of the prompt")
    model_name: str = Field(default="openrouter/gpt-4o", description="The name of the model")
    memory_name: Optional[str] = Field(default=None,
                                       description="Name of the optimizer memory system for recording optimization history")
    clip_ratio: float = Field(default=0.2, description="Clipping ratio for REINFORCE++")
    beta: float = Field(default=0.01, description="KL penalty coefficient")
    reward_fn: Optional[Callable[[str,str,str], Any]] = Field(default=None, description="Custom reward function for evaluating a single candidate")
    batchsize: int = Field(default=10, description="Batch size for aggregating historical reflections")

    def __init__(self,
                 workdir: str,
                 prompt_name: str = "reflection_optimizer",
                 model_name: str = "openrouter/gpt-4o",
                 memory_name: Optional[str] = "optimizer_memory_system",
                 optimize_trainable_variables: bool = True,
                 optimize_solution: bool = True,
                 clip_ratio: float = 0.2,
                 beta: float = 0.01,
                 reward_fn: Optional[Callable[[str,str,str], Any]] = None,
                 batchsize: int = 10,
                 **kwargs
                 ):
        """
        Initialize the REINFORCE++ optimizer.

        Args:
            workdir: Working directory for the optimizer
            prompt_name: Name of the prompt used for optimization
            model_name: Model name for optimization.
            memory_name: Optional name of the optimizer memory system for recording optimization history.
            optimize_trainable_variables: Whether to optimize trainable variables (prompt/tool) in phase 1
            optimize_solution: Whether to optimize solution in phase 2
            reward_fn: Optional custom reward function that takes a single candidate text and returns a reward score.
        """
        super().__init__(
            workdir=workdir,
            prompt_name=prompt_name,
            model_name=model_name,
            memory_name=memory_name,
            **kwargs)
        self.workdir = workdir
        if model_name:
            self.model_name = model_name
        if prompt_name:
            self.prompt_name = prompt_name
        self.memory_name = memory_name

        self.optimize_trainable_variables = optimize_trainable_variables
        self.optimize_solution = optimize_solution

        # REINFORCE++ config
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.reward_fn = reward_fn
        self.batchsize = batchsize

        # Initialize tokenizer for text similarity calculation
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')

    async def _read_historical_reflections(self, results_file_path: str) -> List[str]:
        """
        Read historical phase 1 reflections from results file.

        Args:
            results_file_path: Path to the results JSON file

        Returns:
            List of historical reflection texts (phase 1)
        """
        import json
        historical_reflections = []

        try:
            if not os.path.exists(results_file_path):
                logger.warning(f"Results file not found: {results_file_path}")
                return historical_reflections

            with open(results_file_path, 'r', encoding='utf-8') as f:
                results_data = json.load(f)

            results = results_data.get('results', [])
            for result in reversed(results):  # Start from most recent
                reflection_process = result.get('reflection_process', {})
                phase1_reflections = reflection_process.get('reflection_rounds', [])

                # Extract the last phase 1 reflection from all rounds
                if phase1_reflections:
                    # Find all phase 1 rounds and get the last one
                    phase1_rounds = [r for r in phase1_reflections if r.get('phase') == 1]
                    if phase1_rounds and 'reflection_text' in phase1_rounds[-1]:
                        historical_reflections.append(phase1_rounds[-1]['reflection_text'])

            logger.info(f"Loaded {len(historical_reflections)} historical phase 1 reflections")
            return historical_reflections

        except Exception as e:
            logger.warning(f"Failed to read historical reflections: {e}")
            return historical_reflections

    async def get_trainable_variables(self, agent: Optional[Any] = None) -> Dict[str, Any]:
        """
        Get trainable variables from prompt and tools only.

        Returns:
            Dict[str, Variable]: Dictionary mapping variable names to Variable objects.
        """
        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager
        from src.tool import tcp

        variables: Dict[str, Any] = {}

        # Get trainable variables from prompt (returns Dict[str, Variable])
        if agent and hasattr(agent, 'prompt_name'):
            prompt_name = agent.prompt_name
            prompt_variables_dict = await prompt_manager.get_trainable_variables(prompt_name=prompt_name)
            variables.update(prompt_variables_dict)

        # Get trainable variables from tools (returns Dict[str, Variable])
        tool_variables_dict = await tcp.get_trainable_variables()
        variables.update(tool_variables_dict)

        return variables

    async def _format_variables(self, variables: Dict[str, Any]) -> str:
        """
        Format variables for context (now handles flattened structure).
        
        Args:
            variables (Dict[str, Any]): Dictionary of flattened variables (prompt sub-variables + tool variables).
        """
        
        variables_text = ""
        
        # Step1: Format prompt sub-variables (now flattened, no more nesting)
        prompt_variables_text = "<prompt_variables>\n"
        prompt_variables = {k: v for k, v in variables.items() if isinstance(v, Variable) and (v.type == "system_prompt" or v.type == "agent_message_prompt")}
        for prompt_name, prompt_variable in prompt_variables.items():
            prompt_variables_text += f"<{prompt_name}>\n"
            prompt_variables_text += f"The variable is as follows:\n"
            prompt_variables_text += f"{prompt_variable.variables}"
            prompt_variables_text += f"</{prompt_name}>\n"
        prompt_variables_text += "</prompt_variables>\n"
        variables_text += prompt_variables_text
        
        # Step2: Format tool variables
        tool_variables_text = "<tool_variables>\n"
        tool_variables = {k: v for k, v in variables.items() if isinstance(v, Variable) and v.type == "tool_code"}
        for tool_name, tool_variable in tool_variables.items():
            tool_variables_text += f"<{tool_name}>\n"
            tool_variables_text += f"The variable is as follows:\n"
            tool_variables_text += f"{tool_variable.variables}"
            tool_variables_text += f"</{tool_name}>\n"
        tool_variables_text += "</tool_variables>\n"
        variables_text += tool_variables_text
        
        # Step3: Format solution variable (if present)
        solution_variables = {k: v for k, v in variables.items() if isinstance(v, Variable) and v.type == "solution"}
        if solution_variables:
            solution_variable_text = "<solution_variables>\n"
            for solution_name, solution_variable in solution_variables.items():
                solution_variable_text += f"<{solution_name}>\n"
                solution_variable_text += f"The variable is as follows:\n"
                solution_variable_text += f"{solution_variable.variables}"
                solution_variable_text += f"</{solution_name}>\n"
            solution_variable_text += "</solution_variables>\n"
            variables_text += solution_variable_text
        
        return variables_text

    async def _generate_reflection(self, task: str, variables: Dict[str, Any], execution_result: str,
                                  reward: float, advantage: float, objective: float, policy_ratio: float) -> str:
        """
        Generate the reflection analysis for all variables based on RL metrics.

        Args:
            task (str): Task description.
            variables (Dict[str, Any]): Dictionary of variables.
            execution_result (str): Agent execution result.
            reward (float): Current reward score.
            advantage (float): Calculated advantage.
            objective (float): REINFORCE++ objective value.
            policy_ratio (float): Policy ratio between current and old solution.
        Returns:
            str: Reflection analysis identifying which variables to optimize and how.
        """
        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager

        # Ensure prompt_manager is initialized
        if not hasattr(prompt_manager, 'prompt_context_manager'):
            await prompt_manager.initialize()

        current_variables_text = await self._format_variables(variables)

        system_modules = {}
        agent_message_modules = {
            "task": task,
            "current_variables": current_variables_text,
            "execution_result": execution_result,
            "reward": f"{reward:.4f}",
            "advantage": f"{advantage:.4f}",
            "objective": f"{objective:.4f}",
            "policy_ratio": f"{policy_ratio:.4f}",
        }
        messages = await prompt_manager.get_messages(
            prompt_name=f"{self.prompt_name}_reflection",
            system_modules=system_modules,
            agent_modules=agent_message_modules,
        )

        logger.info(f"| 🤔 Generating reflection analysis using RL metrics (reward={reward:.4f}, advantage={advantage:.4f}, objective={objective:.4f}, ratio={policy_ratio:.4f})...")

        try:
            response = await model_manager(model=self.model_name, messages=messages)
            reflection_text = response.message if hasattr(response, 'message') else str(response)

            logger.info(f"| ✅ Reflection analysis generated ({len(reflection_text)} chars)")
            logger.info(f"| Reflection analysis:\n{reflection_text}\n")

            return reflection_text
        except Exception as e:
            logger.error(f"| ❌ Error generating reflection: {e}")
            raise

    async def _improve_variables(self, task: str, variables: Dict[str, Variable],
                                 reflection_analysis: str, historical_reflections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Improve variables based on reflection analysis. May improve multiple variables simultaneously.
        Uses different optimization logic based on variable types.

        Args:
            task (str): Task description.
            variables: List of Variable objects to potentially improve.
            reflection_analysis (str): Reflection analysis output.
            variable_mapping: Mapping from variable name to Variable object.

        Returns:
            Dictionary of improved variables in flattened structure
            {
                # prompt sub-variables (flattened from system/agent prompts)
                "agent_context_rules": {
                    "name": "agent_context_rules",
                    "variables": "You are a helpful assistant."
                },
                "tool_context_rules": {
                    "name": "tool_context_rules",
                    "variables": "You can use the following tools: {tools}"
                },
                # tool variables
                "bash": {
                    "name": "bash",
                    "variables": "def bash_tool():\n    # tool implementation\n    pass"
                }
            }
        """
        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager

        # Ensure prompt_manager is initialized
        if not hasattr(prompt_manager, 'prompt_context_manager'):
            await prompt_manager.initialize()

        # Format all variables for context
        current_variables_text = await self._format_variables(variables)

        # Combine current reflection with historical reflections if available
        combined_reflection = reflection_analysis
        if historical_reflections:
            logger.info(f"Aggregating {len(historical_reflections)} historical reflections with current reflection for batch processing")
            combined_reflection = f"""
Current Task Reflection:
{reflection_analysis}

Historical Reflections from Previous Tasks:
"""
            for i, hist_reflection in enumerate(historical_reflections):
                combined_reflection += f"\n--- Historical Reflection {i+1} ---\n{hist_reflection}"

            combined_reflection += "\n\nPlease analyze all the above reflections (current and historical) to identify common reasoning patterns and provide more generalizable improvements that work across multiple tasks."
        else:
            logger.info("No historical reflections available, using current reflection only")

        system_modules = {}
        agent_message_modules = {
            "task": task,
            "current_variables": current_variables_text,
            "reflection_analysis": combined_reflection
        }
        messages = await prompt_manager.get_messages(
            prompt_name=f"{self.prompt_name}_improvement",
            system_modules=system_modules,
            agent_modules=agent_message_modules,
        )

        logger.info(f"| ✨ Generating improved variables (may improve multiple variables)...")

        try:
            response = await model_manager(model=self.model_name, messages=messages, response_format=ImprovedVariables)
            improved_variables: ImprovedVariables = response.extra.parsed_model
            variables = {
                variable.name: variable.model_dump() for variable in improved_variables.variables
            }
            return variables
        except Exception as e:
            logger.error(f"| ❌ Error improving variables: {e}")
            raise

    async def _improve_solution(self, task: str, variables: Dict[str, Variable],
                                 reflection_analysis: str) -> Response:

        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager

        # Ensure prompt_manager is initialized
        if not hasattr(prompt_manager, 'prompt_context_manager'):
            await prompt_manager.initialize()

        # Format all variables for context
        current_variables_text = await self._format_variables(variables)

        system_modules = {}
        agent_message_modules = {
            "task": task,
            "current_variables": current_variables_text,
            "reflection_analysis": reflection_analysis
        }
        messages = await prompt_manager.get_messages(
            prompt_name=f"{self.prompt_name}_improvement",
            system_modules=system_modules,
            agent_modules=agent_message_modules,
        )

        logger.info(f"| ✨ Generating improved solution")

        try:
            response = await model_manager(model=self.model_name, messages=messages, response_format=Response)
            improved_solution: Response = response.extra.parsed_model
            return improved_solution
        except Exception as e:
            logger.error(f"| ❌ Error improving solution: {e}")
            raise

    def _calculate_advantage(self, reward: float, kl_penalty: float) -> float:
        """
        Calculate advantage using REINFORCE++ formula.
        A(s, a) = reward - KL_penalty
        """
        advantage = reward - kl_penalty
        return advantage

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into tokens for edit distance calculation.
        """
        token_ids = self.tokenizer.encode(text)
        # Use token ids as strings for comparison to avoid decoding overhead
        return [str(tid) for tid in token_ids]

    def _levenshtein_distance(self, tokens1: List[str], tokens2: List[str]) -> int:
        """
        Calculate Levenshtein edit distance between two token sequences.

        :param tokens1: First token sequence
        :param tokens2: Second token sequence
        :return: Edit distance (minimum number of insertions, deletions, or substitutions)
        """
        m, n = len(tokens1), len(tokens2)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize: empty sequence to tokens2
        for j in range(n + 1):
            dp[0][j] = j

        # Initialize: tokens1 to empty sequence
        for i in range(m + 1):
            dp[i][0] = i

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    # Tokens match, no operation needed
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Take minimum of three operations
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,  # Delete token from tokens1
                        dp[i][j - 1] + 1,  # Insert token from tokens2
                        dp[i - 1][j - 1] + 1  # Substitute token
                    )

        return dp[m][n]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using token-level Levenshtein edit distance.
        """
        if not text1 or not text2:
            return 0.0

        tokens1 = self._tokenize_text(text1)
        tokens2 = self._tokenize_text(text2)

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        distance = self._levenshtein_distance(tokens1, tokens2)
        max_len = max(len(tokens1), len(tokens2))
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)

    def _apply_clipping(self, policy_ratio: float) -> float:
        """
        Apply REINFORCE++ clipping to a policy ratio.
        Clip ratio to [1-ε, 1+ε] where ε = clip_ratio
        """
        return max(1 - self.clip_ratio, min(1 + self.clip_ratio, policy_ratio))

    def _calculate_reinforce_plus_plus_objective(self, policy_ratio: float, clipped_ratio: float, advantage: float) -> float:
        """
        Calculate REINFORCE++ objective with clipping for a single sample.
        L^CLIP(θ) = min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)
        """
        return min(clipped_ratio * advantage, policy_ratio * advantage)

    async def _evaluate_solution(self, task: str, execution_result: str) -> EvaluationResult:
        """
        Evaluate if the current solution is satisfactory.
        """
        logger.info(f"| ⚖️ Evaluating if optimization goal is reached...")

        system_prompt = dedent(f"""
            You are an expert evaluator. Your task is to determine if the current agent solution and reasoning have successfully completed the given task.

            Task: {task}

            Review the current solution and reasoning provided below and decide if the optimization process can stop.
            Evaluate both the correctness of the content and the format compliance. The solution must be correct, complete, follow all requirements, and be in the proper format that can be correctly parsed by the evaluation system.
            If the solution is correct, complete, follows all requirements, and is properly formatted, set is_satisfied to True.
            Otherwise, set is_satisfied to False and provide the reason.

            Return your decision in the specified structured format.
            """)

        user_message = dedent(f"""
            Current Execution Result/Solution:
            {execution_result}

            Please evaluate if this solution is satisfactory.
            """)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        try:
            response = await model_manager(model=self.model_name, messages=messages, response_format=EvaluationResult)
            evaluation: EvaluationResult = response.extra.parsed_model
            logger.info(f"| Evaluation result: Satisfied={evaluation.is_satisfied}, Reason: {evaluation.reason}")
            return evaluation
        except Exception as e:
            logger.warning(f"| ⚠️ Evaluation failed: {e}. Optimization will continue...")
            return EvaluationResult(is_satisfied=False, reason=f"Evaluation failed: {e}")

    async def optimize(
            self,
            agent: Any,
            task: str,
            ground_truth: str,
            sft_solution: str,
            files: Optional[List[str]] = None,
            results_file_path: Optional[str] = None,
            ctx: "SessionContext" = None,
            **kwargs
    ):
        """
        Optimize the agent prompt using the Reflection approach.

        Args:
            agent: Agent instance.
            task: Task description to optimize for.
            ground_truth: Ground truth for evaluation.
            sft_solution: SFT solution for policy ratio calculation.
            files: Optional list of attachments.
            results_file_path: Optional path to results file for historical reflections.
            ctx: Session context.
        """

        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager
        from src.tool import tcp
        from src.environment import ecp
        from src.agent import acp
        from src.memory import memory_manager
        from src.session import SessionContext
        from src.memory.types import EventType

        if ctx is None:
            ctx = SessionContext()
        id = ctx.id
        memory_ctx = SessionContext(id=id)
        agent_ctx = SessionContext(id=id)

        # Use optimization_steps if provided, otherwise use self.max_steps
        optimization_steps = self.max_steps

        # Initialize optimizer memory session if available
        memory_name = self.memory_name
        task_id = None
        if memory_name:
            try:
                agent_name = getattr(agent, 'name', 'unknown_agent')
                task_id = f"opt_task_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                await memory_manager.start_session(
                    memory_name=memory_name,
                    ctx=memory_ctx
                )

                # Add optimization task start event
                await memory_manager.add_event(
                    memory_name=memory_name,
                    step_number=0,
                    event_type=EventType.TASK_START,
                    data=dict(
                        task=task,
                        optimization_steps=optimization_steps,
                        optimize_trainable_variables=self.optimize_trainable_variables,
                        optimize_solution=self.optimize_solution
                    ),
                    agent_name=agent_name,
                    task_id=task_id,
                    ctx=memory_ctx
                )
            except Exception as e:
                logger.warning(f"| ⚠️ Failed to initialize optimizer memory: {e}")
                memory_name = None
                task_id = None

        # Run agent once to get initial solution
        logger.info(f"| 🚀 Running agent to get initial solution...")
        agent_response = await agent(task=task, files=files, ctx=agent_ctx)
        agent_response_extra_data = agent_response.extra.data if agent_response.extra and agent_response.extra.data else None
        current_agent_result = agent_response_extra_data['result']
        current_agent_reasoning = agent_response_extra_data['reasoning']
        current_solution = f"Result: {current_agent_result}\nReasoning: {current_agent_reasoning}" if current_agent_reasoning else f"Result: {current_agent_result}"
        logger.info(f"| ✅ Initial solution obtained")

        old_solution = current_solution
        # Run the optimization loop.
        for opt_step in range(optimization_steps):
            logger.info(f"| {'=' * 60}")
            logger.info(f"| REINFORCE++ Optimization Step {opt_step + 1}/{optimization_steps}")
            logger.info(f"| {'=' * 60}")

            current_reward = await self.reward_fn(answer=current_agent_result, ground_truth=ground_truth)
            logger.info(f"| 📊 Current reward: {current_reward}")

            initial_policy_ratio = self._calculate_text_similarity(sft_solution, current_solution)
            kl_div_candidate = abs(math.log(max(initial_policy_ratio, 1e-8)))
            kl_penalty_candidate = self.beta * kl_div_candidate

            # Calculate advantage for candidate: A = reward - KL_penalty
            advantage_candidate = current_reward - kl_penalty_candidate

            # Calculate policy ratio for candidate
            policy_ratio = self._calculate_text_similarity(old_solution, current_solution)

            # Apply clipping and calculate objective for candidate
            clipped_ratio = self._apply_clipping(policy_ratio)
            objective_current = self._calculate_reinforce_plus_plus_objective(policy_ratio,
                                                                                clipped_ratio,
                                                                                advantage_candidate)
            try:
                # ============ PHASE 1: Optimize Trainable Variables ============
                if self.optimize_trainable_variables:
                    logger.info(f"| 🔧 Phase 1: Optimizing trainable variables (prompt/tool)...")

                    # Get trainable variables
                    trainable_variables = await self.get_trainable_variables(agent=agent)

                    if len(trainable_variables) > 0:
                        # Generate reflection analysis for trainable variables using RL metrics
                        reflection_analysis = await self._generate_reflection(
                            task=task,
                            variables=trainable_variables,
                            execution_result=current_solution,
                            reward=current_reward,
                            advantage=advantage_candidate,
                            policy_ratio=policy_ratio,
                            objective=objective_current,
                        )

                        # Read historical reflections and improve trainable variables
                        historical_reflections = None
                        if results_file_path:
                            historical_reflections = await self._read_historical_reflections(results_file_path)
                            # Take only the most recent batchsize-1 reflections
                            if len(historical_reflections) >= self.batchsize - 1:
                                historical_reflections = historical_reflections[:(self.batchsize - 1)]

                        improved_variables = await self._improve_variables(
                            task=task,
                            variables=trainable_variables,
                            reflection_analysis=reflection_analysis,
                            historical_reflections=historical_reflections,
                        )

                        # Update trainable variables (now flattened structure)
                        # Group prompt sub-variables together for batch update
                        prompt_updates = {}  # Will collect all prompt sub-variable updates
                        variables_updated = False

                        for variable_name, improved_var in improved_variables.items():
                            if variable_name not in trainable_variables:
                                logger.warning(
                                    f"| ⚠️ Variable {variable_name} not found in trainable variables, skipping")
                                continue

                            variable_type = trainable_variables[variable_name].type
                            # Extract the actual value string from ImprovedVariable dict
                            variable_value = improved_var['variables']

                            if variable_type == "system_prompt" or variable_type == "agent_message_prompt":
                                # Prompt sub-variables - collect for batch update
                                prompt_updates[variable_name] = variable_value
                                logger.debug(f"| 📝 Collected prompt sub-variable update: {variable_name}")
                            elif variable_type == "tool_code":
                                # tcp.set_variables expects {"name": tool_name, "variables": code_string}
                                tool_variable_updates = {"name": variable_name, "variables": variable_value}
                                await tcp.set_variables(tool_name=variable_name, variable_updates=tool_variable_updates)
                                variables_updated = True
                                logger.info(f"| ✅ Updated tool variable: {variable_name}")
                            elif variable_type == "environment_code":
                                # ecp.set_variables expects {"name": env_name, "variables": code_string}
                                env_variable_updates = {"name": variable_name, "variables": variable_value}
                                await ecp.set_variables(env_name=variable_name, variable_updates=env_variable_updates)
                                variables_updated = True
                                logger.info(f"| ✅ Updated environment variable: {variable_name}")
                            elif variable_type == "agent_code":
                                # acp.set_variables expects {"name": agent_name, "variables": code_string}
                                agent_variable_updates = {"name": variable_name, "variables": variable_value}
                                await acp.set_variables(agent_name=variable_name, variable_updates=agent_variable_updates)
                                variables_updated = True
                                logger.info(f"| ✅ Updated agent variable: {variable_name}")
                            elif variable_type == "memory_code":
                                # memory_manager.set_variables expects {"name": memory_name, "variables": code_string}
                                memory_variable_updates = {"name": variable_name, "variables": variable_value}
                                await memory_manager.set_variables(memory_name=variable_name,
                                                                   variable_updates=memory_variable_updates)
                                variables_updated = True
                                logger.info(f"| ✅ Updated memory variable: {variable_name}")

                        # Batch update all prompt sub-variables
                        if prompt_updates:
                            prompt_name = agent.prompt_name if hasattr(agent, 'prompt_name') else "tool_calling"
                            await prompt_manager.set_variables(
                                prompt_name=prompt_name,
                                variable_updates=prompt_updates
                            )
                            variables_updated = True
                            logger.info(
                                f"| ✅ Updated {len(prompt_updates)} prompt sub-variables: {list(prompt_updates.keys())}")

                        if variables_updated:
                            # Re-run agent with updated variables
                            logger.info(f"| 🔄 Re-running agent with updated trainable variables...")
                            agent_response = await agent(task=task, files=files, ctx=agent_ctx)
                            agent_response_extra_data = agent_response.extra.data if agent_response.extra and agent_response.extra.data else None
                            current_agent_result = agent_response_extra_data['result']
                            current_agent_reasoning = agent_response_extra_data['reasoning']
                            current_solution = f"Result: {current_agent_result}\nReasoning: {current_agent_reasoning}" if current_agent_reasoning else f"Result: {current_agent_result}"
                            logger.info(f"| ✅ Phase 1 completed - trainable variables updated")
                        else:
                            logger.info(f"| ℹ️ Phase 1: No trainable variables were updated")

                        # Record phase 1 to memory
                        if memory_name and variables_updated:
                            try:
                                event_data = {
                                    "phase": "trainable_variables",
                                    "task": task,
                                    "reflection_analysis": reflection_analysis,
                                    "execution_result": current_solution,
                                    "variable_changes": {}
                                }

                                for var_name, improved_var in improved_variables.items():
                                    if var_name in trainable_variables:
                                        before_var = trainable_variables[var_name]
                                        before_value = before_var.get_value() if hasattr(before_var,
                                                                                         'get_value') else str(
                                            before_var.variables)
                                        # Extract the actual value from ImprovedVariable dict
                                        after_value = improved_var['variables'] if isinstance(improved_var,
                                                                                        dict) else str(
                                            improved_var)

                                        event_data["variable_changes"][var_name] = {
                                            "type": before_var.type,
                                            "before": before_value,
                                            "after": after_value
                                        }

                                await memory_manager.add_event(
                                    memory_name=memory_name,
                                    step_number=opt_step * 2 + 1,
                                    event_type=EventType.OPTIMIZATION_STEP,
                                    data=event_data,
                                    agent_name=getattr(agent, 'name', 'unknown_agent'),
                                    task_id=task_id,
                                    ctx=memory_ctx
                                )
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to record phase 1 to memory: {e}")
                    else:
                        logger.info(f"| ℹ️ Phase 1: No trainable variables found, skipping")
                else:
                    logger.info(f"| ⏭️ Phase 1: Skipped (optimize_trainable_variables=False)")

                # ============ PHASE 2: Optimize Solution ============
                if self.optimize_solution:
                    logger.info(f"| 📝 Phase 2: Optimizing solution...")

                    # Create solution variable for optimization
                    solution_variable = Variable(
                        name="solution",
                        type="solution",
                        description="The current solution that may need improvement.",
                        require_grad=True,
                        variables=current_solution
                    )

                    solution_variables = {"solution": solution_variable}

                    # Generate reflection for solution using RL metrics
                    solution_reflection = await self._generate_reflection(
                        task=task,
                        variables=solution_variables,
                        execution_result=current_solution,
                        reward=current_reward,
                        advantage=advantage_candidate,
                        policy_ratio=policy_ratio,
                        objective=objective_current,
                    )

                    # Improve solution based on reflection
                    improved_solution_result = await self._improve_solution(
                        task=task,
                        variables=solution_variables,
                        reflection_analysis=solution_reflection,
                    )

                    # Check if solution was improved
                    if improved_solution_result.result:
                        current_agent_result = improved_solution_result.result
                        current_agent_reasoning = improved_solution_result.reasoning
                        current_solution = f"Result: {current_agent_result}\nReasoning: {current_agent_reasoning}" if current_agent_reasoning else f"Result: {current_agent_result}"

                        logger.info(f"| ✅ Phase 2 completed - solution optimized")

                        # Record phase 2 to memory
                        if memory_name:
                            try:
                                event_data = {
                                    "phase": "solution",
                                    "task": task,
                                    "reflection_analysis": solution_reflection,
                                    "before_solution": solution_variable.get_value(),
                                    "after_solution": current_solution
                                }

                                await memory_manager.add_event(
                                    memory_name=memory_name,
                                    step_number=opt_step * 2 + 2,
                                    event_type=EventType.OPTIMIZATION_STEP,
                                    data=event_data,
                                    agent_name=getattr(agent, 'name', 'unknown_agent'),
                                    task_id=task_id,
                                    ctx=memory_ctx
                                )
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to record phase 2 to memory: {e}")
                    else:
                        logger.info(f"| ℹ️ Phase 2: No solution improvements suggested")
                else:
                    logger.info(f"| ⏭️ Phase 2: Skipped (optimize_solution=False)")

                # ============ Evaluation Module: Check for early termination ============
                evaluation = await self._evaluate_solution(
                    task=task,
                    execution_result=current_solution
                )

                if evaluation.is_satisfied:
                    logger.info(f"| 🎉 Early termination triggered: {evaluation.reason}")
                    break

                logger.info(f"| ✅ Optimization step {opt_step + 1} completed\n")

            except Exception as e:
                logger.error(f"| ❌ Error in optimization step {opt_step + 1}: {e}")
                import traceback
                logger.error(f"| Traceback: {traceback.format_exc()}")
                # Continue with the next iteration.
                continue

        # End optimization memory session if available
        if memory_name:
            try:
                # Add optimization task end event
                await memory_manager.add_event(
                    memory_name=memory_name,
                    step_number=optimization_steps * 2 + 1,
                    event_type=EventType.TASK_END,
                    data=dict(
                        task=task,
                        optimization_steps=optimization_steps,
                        completed=True,
                        final_solution=current_solution
                    ),
                    agent_name=getattr(agent, 'name', 'unknown_agent'),
                    task_id=task_id,
                    ctx=memory_ctx
                )

                await memory_manager.end_session(memory_name=memory_name, ctx=memory_ctx)
                logger.info(f"| 📝 Ended optimization memory session: {id}")
            except Exception as e:
                logger.warning(f"| ⚠️ Failed to end optimization memory session: {e}")

        logger.info(f"\n| {'=' * 60}")
        logger.info(f"| ✅ Rinforce++ optimization completed!")
        logger.info(f"| {'=' * 60}")
        logger.info(f"| Final solution:\n{current_solution}")
        logger.info(f"| {'=' * 60}")

        return current_agent_reasoning, current_agent_result
