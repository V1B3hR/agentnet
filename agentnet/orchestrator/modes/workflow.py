"""
Enhanced, asynchronous Workflow strategy implementation.

This strategy transforms a high-level task into a structured plan and then
executes that plan step-by-step, carrying context between steps and handling
both tool-based and reasoning-based actions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.enums import Mode, ProblemSolvingStyle, ProblemTechnique
from .base import BaseStrategy

if TYPE_CHECKING:
    from ...core.agent import AgentNet


class WorkflowStrategy(BaseStrategy):
    """
    An advanced strategy that first plans a sequence of steps and then executes
    them sequentially to achieve a complex goal.

    The process involves:
    1. Planning Phase: An agent breaks the task into a structured JSON plan.
    2. Execution Phase: The strategy iterates through the plan, executing each
       step using either agent reasoning or a specified tool, and carries the
       context from one step to the next.
    """

    def __init__(
        self,
        style: Optional[ProblemSolvingStyle] = ProblemSolvingStyle.IMPLEMENTOR,
        technique: Optional[ProblemTechnique] = None,
        **config: Any,
    ):
        """
        Initialize the workflow strategy.

        Args:
            style: The default style is 'implementor' for its focus on execution.
            technique: Optional problem-solving technique.
            **config: Configuration for the strategy, e.g.,
                - max_steps (int): A safeguard to limit the number of steps. Defaults to 10.
        """
        super().__init__(Mode.WORKFLOW, style, technique, **config)

    async def _execute(
        self,
        agent: "AgentNet",
        task: str,
        context: Dict[str, Any],
        agents: Optional[List["AgentNet"]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the planning and execution phases of the workflow.
        """
        max_steps = self.config.get("max_steps", 10)

        # --- Phase 1: Planning ---
        self.logger.info(f"Phase 1: Generating workflow plan for task: {task[:100]}...")
        plan = await self._generate_plan(agent, task, context)
        
        if not plan or len(plan) > max_steps:
            raise ValueError(f"Failed to generate a valid plan or plan exceeds max_steps ({max_steps}).")

        self.logger.info(f"Plan generated with {len(plan)} steps. Starting execution.")

        # --- Phase 2: Execution ---
        execution_trace = []
        for i, step in enumerate(plan):
            step_name = step.get("step_name", f"step_{i+1}")
            self.logger.info(f"Executing Step {i+1}/{len(plan)}: {step_name}")

            try:
                # The state contains results from all previous steps
                previous_steps_context = self.get_state("previous_steps_context", {})
                
                if "tool_name" in step and step["tool_name"]:
                    result = await agent.execute_tool(
                        tool_name=step["tool_name"],
                        parameters=step.get("parameters", {}),
                        context=previous_steps_context,
                    )
                    if not result or result.get("status") != "success":
                        raise RuntimeError(f"Tool '{step['tool_name']}' failed: {result.get('error_message', 'Unknown error')}")
                    step_output = result.get("data")
                else:
                    step_prompt = self._create_step_prompt(task, step, previous_steps_context)
                    result = await agent.async_generate_reasoning_tree(
                        task=step_prompt,
                        confidence_threshold=0.8,
                        metadata={"workflow_step": i + 1, "step_name": step_name},
                    )
                    step_output = result.get("result", {}).get("content")

                # --- Update State and Trace ---
                execution_trace.append({"step": step_name, "status": "success", "output": step_output})
                previous_steps_context[step_name] = step_output
                self.update_state("previous_steps_context", previous_steps_context)

            except Exception as e:
                self.logger.error(f"Workflow failed at step '{step_name}': {e}")
                execution_trace.append({"step": step_name, "status": "failed", "error": str(e)})
                # Re-raise to be caught by the base strategy's run method
                raise RuntimeError(f"Workflow failed at step '{step_name}'. See trace for details.") from e

        return {
            "initial_plan": plan,
            "execution_trace": execution_trace,
            "final_output": execution_trace[-1].get("output") if execution_trace else None,
        }

    async def _generate_plan(self, agent: "AgentNet", task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prompts an agent to create a structured, step-by-step plan in JSON format."""
        
        tools_list = agent.list_available_tools()
        tools_str = json.dumps(tools_list, indent=2) if tools_list else "No tools available."

        planning_prompt = (
            f"You are a meticulous workflow planner. Your task is to break down the following high-level goal into a sequence of concrete, executable steps. You must respond with only a valid JSON array.\n\n"
            f"**High-Level Goal:** {task}\n\n"
            f"**Available Tools:**\n{tools_str}\n\n"
            f"**Instructions:**\n"
            f"1. Think step-by-step to achieve the goal.\n"
            f"2. For each step, define a unique `step_name`.\n"
            f"3. If a tool is the best way to perform a step, specify the `tool_name` and the required `parameters`.\n"
            f"4. If a step requires reasoning or summarization, omit the `tool_name` and provide a clear `description` of the task for another AI agent.\n"
            f"5. Ensure the output is a single, valid JSON array of step objects.\n\n"
            f"**JSON Object Schema:**\n"
            f"{{ \"step_name\": string, \"description\": string, \"tool_name\": Optional[string], \"parameters\": Optional[Dict[string, any]] }}\n\n"
            f"Begin."
        )

        plan_result = await agent.async_generate_reasoning_tree(
            task=planning_prompt,
            confidence_threshold=0.8,
            metadata={"workflow_phase": "planning"},
        )
        
        plan_content = plan_result.get("result", {}).get("content", "[]")
        try:
            # Clean the content to extract only the JSON part
            json_match = re.search(r'\[.*\]', plan_content, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON array found in the response.", plan_content, 0)
            
            return json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode plan from agent response: {e}\nResponse was:\n{plan_content}")
            return []

    def _create_step_prompt(self, original_task: str, step: Dict, context: Dict) -> str:
        """Creates a prompt for a reasoning-based workflow step."""
        context_str = "\n".join(f" - Result of '{name}': {str(res)[:200]}..." for name, res in context.items())
        
        return (
            f"**Original Goal
