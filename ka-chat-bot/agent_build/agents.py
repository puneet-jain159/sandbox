from __future__ import annotations

"""Agent implementations for the procurement workflow."""
from typing import Any, Dict, Optional, TypedDict, Annotated
from agent_build.utils import add_message_if_not_exists

from langchain_core.messages import AnyMessage
from langchain.schema import HumanMessage
from langgraph.graph import END
from langgraph.graph.message import add_messages

# Types
class AgentState(TypedDict):
    original_prompt: str
    user_confirmed_hierarchy: Optional[str]
    extracted_material: Optional[str]
    extracted_location: Optional[str]
    user_confirmed_location_hierarchy: Optional[str]
    combined_prompt: Optional[str]
    final_response: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    worker_outputs: Optional[Dict[str, str]]
    next_node: Optional[str]


def material_hierarchy_resolver_agent(state: AgentState, llm_model, logger, default_config, utils) -> Dict[str, Any]:
    _ = logger
    if not llm_model:
        base_messages = state.get("messages", [])
        new_msg = {"role": "system", "content": default_config.messages.llm_unavailable}
        return {
            "extracted_material": None,
            "messages": add_message_if_not_exists(base_messages, new_msg),
        }

    prompt = state["original_prompt"]
    messages = state.get("messages", [])
    user_confirmed_hierarchy = state.get("user_confirmed_hierarchy", [])
    user_confirmed_material_name = state.get("extracted_material", None)

    if user_confirmed_hierarchy is None:
        extraction_prompt = default_config.material_prompt.build_prompt(prompt)
        llm_response = llm_model.invoke(extraction_prompt)
        llm_output = llm_response.content.strip()

        extraction_result = utils.extract_llm_response(llm_output, ["Material", "Hierarchy"])
        extracted_material = extraction_result.get("Material")
        extracted_hierarchy = extraction_result.get("Hierarchy")

        confirmed_hierarchy, combined_prompt, updated_messages = utils.process_extraction_results(
            extracted_material,
            extracted_hierarchy,
            "material",
            prompt,
            messages,
            utils.call_get_material_hierarchy_level,
            f"Original request: '{prompt}'",
        )
        messages = updated_messages
    else:
        if user_confirmed_hierarchy == "skip":
            confirmed_hierarchy = None
            extracted_material = None
        else:
            confirmed_hierarchy = user_confirmed_hierarchy
            extracted_material = (
                user_confirmed_material_name if user_confirmed_material_name != [] else None
            )

        if confirmed_hierarchy:
            combined_prompt = (
                f"Original request: '{prompt}'\nHierarchy level for {extracted_material} is: '{confirmed_hierarchy}'"
            )
            messages = add_message_if_not_exists(
                messages,
                {
                    "role": "assistant",
                    "content": f"Confirmed hierarchy is '{confirmed_hierarchy}' for the material '{extracted_material}'.",
                },
            )
        else:
            combined_prompt = f"Original request: '{prompt}'"
            messages = add_message_if_not_exists(
                messages,
                {
                    "role": "assistant",
                    "content": "User indicated no specific hierarchy even if there is hierarchy level ambiguity identified",
                },
            )

    state.setdefault("worker_outputs", {})
    state["worker_outputs"]["material_hierarchy_resolver_agent"] = {
        "combined_prompt": combined_prompt
    }

    return {
        "user_confirmed_hierarchy": confirmed_hierarchy,
        "extracted_material": extracted_material,
        "combined_prompt": combined_prompt,
        "messages": messages,
    }


def location_hierarchy_resolver_agent(state: AgentState, llm_model, logger, default_config, utils) -> Dict[str, Any]:
    _ = logger
    if not llm_model:
        base_messages = state.get("messages", [])
        new_msg = {"role": "assistant", "content": default_config.messages.llm_unavailable}
        return {
            "extracted_location": None,
            "messages": add_message_if_not_exists(base_messages, new_msg),
        }

    prompt = state["original_prompt"]
    combined_prompt = state.get("combined_prompt", "")
    messages = state.get("messages", [])
    user_confirmed_location_hierarchy = state.get("user_confirmed_location_hierarchy", None)
    user_confirmed_extracted_location = state.get("extracted_location", None)

    if user_confirmed_location_hierarchy is None:
        extraction_prompt = default_config.location_prompt.build_prompt(prompt)
        llm_response = llm_model.invoke(extraction_prompt)
        llm_output = llm_response.content.strip()
        extraction_result = utils.extract_llm_response(llm_output, ["location", "hierarchy"])
        extracted_location = extraction_result.get("location")
        extracted_hierarchy = extraction_result.get("hierarchy")
        confirmed_location_hierarchy, combined_prompt, updated_messages = utils.process_extraction_results(
            extracted_location,
            extracted_hierarchy,
            "location",
            prompt,
            messages,
            utils.call_get_location_hierarchy_level,
            combined_prompt,
        )
        messages = updated_messages
    else:
        confirmed_location_hierarchy = user_confirmed_location_hierarchy
        extracted_location = (
            user_confirmed_extracted_location if user_confirmed_extracted_location else None
        )
        if confirmed_location_hierarchy == "skip":
            confirmed_location_hierarchy = None
            extracted_location = None
            messages = add_message_if_not_exists(
                messages,
                {"role": "user", "content": "User skipped hierarchy confirmation."},
            )
        else:
            combined_prompt = (
                f"{combined_prompt}\nHierarchy level for {extracted_location}: '{confirmed_location_hierarchy}'"
            )
            messages = add_message_if_not_exists(
                messages,
                {
                    "role": "user",
                    "content": f"Confirmed hierarchy is '{confirmed_location_hierarchy}' for location '{extracted_location}'.",
                },
            )

    return {
        "user_confirmed_location_hierarchy": confirmed_location_hierarchy,
        "extracted_location": extracted_location,
        "combined_prompt": combined_prompt,
        "messages": messages,
    }


def p2p_spend_genie_agent(state: AgentState, genie_agent, default_config, utils, logger) -> Dict[str, Any]:
    _ = logger
    combined_prompt = state.get("combined_prompt")
    messages = state.get("messages", [])
    worker_outputs = state.get("worker_outputs", {})

    is_valid, validation_error = utils.validate_genie_input(combined_prompt, messages)
    if not is_valid:
        error_msg = default_config.messages.format_message(
            default_config.messages.genie_input_validation, validation_error=validation_error
        )
        messages = add_message_if_not_exists(messages, {"role": "assistant", "content": error_msg})
        return {"messages": messages}

    try:
        result = genie_agent.invoke({"messages": [HumanMessage(content=combined_prompt)]})
        response_content, message_dict = utils.process_genie_response(result, "p2p_spend_genie_agent")
        messages = add_message_if_not_exists(messages, message_dict)
        worker_outputs = utils.update_worker_outputs(
            state,
            "p2p_spend_genie_agent",
            {
                "combined_prompt": combined_prompt,
                "response_content": response_content,
                "timestamp": utils.datetime.now().isoformat() if hasattr(utils, "datetime") else "",
            },
        )
        return {"messages": messages, "worker_outputs": worker_outputs}
    except Exception as e:  # noqa: BLE001
        return utils.create_error_response(
            error=e, agent_name="P2P Spend Genie", messages=messages, worker_outputs=worker_outputs
        )


def summary_agent(state: AgentState, llm_model, default_config, logger) -> Dict[str, Any]:
    _ = logger
    messages = state.get("messages", [])
    summary_prompt = default_config.summary_prompt.build_prompt(messages)
    summary_response = llm_model.invoke(summary_prompt).content
    messages = add_message_if_not_exists(messages, {"role": "assistant", "content": f"Summary: '{summary_response}'."})
    state.setdefault("worker_outputs", {})
    state["worker_outputs"]["summary_agent"] = {"summary_response": summary_response}
    return {"summary_response": summary_response, "messages": messages, "next": END}


def supervisor_agent(state: AgentState, llm_model, logger, default_config) -> Dict[str, Any]:
    _ = logger
    allowed_steps = [
        "material_hierarchy_resolver_agent",
        "location_hierarchy_resolver_agent",
        "p2p_spend_genie_agent",
        "summary_agent",
    ]
    worker_outputs = state.get("worker_outputs", {})
    agents_called = set(worker_outputs.keys())
    if "summary_agent" in agents_called:
        return {"next_node": END}

    remaining_steps = [agent for agent in allowed_steps if agent not in agents_called]
    if not llm_model:
        return {"next_node": remaining_steps[0] if remaining_steps else "summary_agent"}

    # Build supervisor decision prompt via config
    prompt = default_config.supervisor_prompt.build_prompt(state, state.get("messages", []))
    try:
        llm_response = llm_model.invoke(prompt)
        next_step_raw = llm_response.content.strip().strip("'").strip('"')
        if next_step_raw in allowed_steps:
            if next_step_raw in agents_called:
                for step in allowed_steps:
                    if step not in agents_called:
                        return {"next_node": step}
                return {"next_node": "summary_agent"}
            return {"next_node": next_step_raw}
        return {"next_node": remaining_steps[0] if remaining_steps else "summary_agent"}
    except Exception:  # noqa: BLE001
        return {"next_node": remaining_steps[0] if remaining_steps else "summary_agent"}

