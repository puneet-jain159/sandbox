from typing import Dict, Any, Optional, List
import uuid
import re
import os
import mlflow
from mlflow.pyfunc import ChatAgent  # type: ignore
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext  # type: ignore
from agent_build.utils import add_message_if_not_exists



# -----------------------------
# Regex Patterns for Extraction
# -----------------------------
_MATERIAL_PATTERN = re.compile(
    r"""
    (?:
        LLM\s+attempted\s+material\s+extraction:\s*Found\s+['"]?(?P<word>[^'"\s]+)
    )
    |
    (?:
        ['"]?(?P<word_alt>[^'"\s]+)['"]?\s+appears\s+in\s+multiple\s+material\s+hierarchies
    )
    """,
    re.IGNORECASE | re.VERBOSE
)
_LOCATION_PATTERN = re.compile(
    r"""
    (?:
        LLM\s+attempted\s+material\s+extraction:\s*Found\s+['"]?(?P<word>[^'"\s]+)
    )
    |
    (?:
        ['"]?(?P<word_alt>[^'"\s]+)['"]?\s+appears\s+in\s+multiple\s+location\s+hierarchies
    )
    """,
    re.IGNORECASE | re.VERBOSE
)



# -----------------------------
# Utility Extraction Functions
# -----------------------------
def extract_material(text: str) -> str | None:
    """Extract material name from LLM log text."""
    match = _MATERIAL_PATTERN.search(text)
    if not match:
        return None
    # Handle both variants: Found 'X' ... OR 'X' appears in multiple material hierarchies
    return match.group("word") if match.groupdict().get("word") else match.group("word_alt")


def extract_location(text: str) -> str | None:
    """Extract material name from LLM log text."""
    match = _LOCATION_PATTERN.search(text)
    if not match:
        return None
    # Handle both variants: Found 'X' ... OR 'X' appears in multiple material hierarchies
    return match.group("word") if match.groupdict().get("word") else match.group("word_alt")


def find_trigger_message(request: dict, trigger_phrase: str, lookback: int = 5) -> str | None:
    """
    Look back through the last `lookback` messages in the request.
    Return the message content if it contains the trigger phrase.
    """
    for message in reversed(request.get('messages', [])[-lookback:]):
        if trigger_phrase in message.get('content', ''):
            return message['content']
    return None


def check_if_heirarchy_resolver_exists(request: dict) -> bool:
    """
    Check if recent messages indicate a material hierarchy resolution step.
    """
    return find_trigger_message(request, "Please confirm the material hierarchy level") is not None


def check_if_location_heirarchy_resolver_exists(request: dict) -> bool:
    """
    Check if recent messages indicate a location hierarchy resolution step.
    """
    return find_trigger_message(request, "Please confirm the location hierarchy level") is not None


# Using mlflow ChatAgent types to enable model logging and deployment


class LangGraphChatAgent(ChatAgent):
    """Adapter that wraps a LangGraph compiled app to the MLflow ChatAgent API.

    Expects an app with methods: stream(state, config, stream_mode), and optionally
    get_state(thread_config) and update_state(thread_config, updates, node_id).
    """

    def __init__(self, app_or_workflow: Optional[Any] = None):
        # Accept either a compiled app (with stream) or a StateGraph-like workflow (with compile)
        self.app = None
        # If not provided, fetch workflow from compile.py
        if app_or_workflow is None:
            try:
                from agent_build.compile import workflow as _default_workflow  # lazy import
                app_or_workflow = _default_workflow
            except Exception as _e:  # noqa: F841
                raise RuntimeError("Failed to import workflow from agent_build.compile")
        if hasattr(app_or_workflow, "stream"):
            self.app = app_or_workflow
        elif hasattr(app_or_workflow, "compile"):
            # Compile with checkpointer strictly; require Postgres when available
            from agent_build.compile import get_pg_checkpointer  # lazy import to avoid cycles
            with get_pg_checkpointer() as checkpointer:  # type: ignore[misc]
                self.app = app_or_workflow.compile(checkpointer=checkpointer)
        else:
            raise TypeError("Expected a compiled app with 'stream' or a workflow with 'compile'.")

    def _convert_messages_to_dict(self, messages: List[ChatAgentMessage]) -> List[Dict[str, Any]]:
        # If MLflow ChatAgentMessage objects are present, convert to dict; else assume dict already
        result: List[Dict[str, Any]] = []
        for m in messages:
            if isinstance(m, dict):
                result.append(m)
            else:
                # Best-effort extraction
                result.append({k: getattr(m, k, None) for k in ("id", "role", "content") if hasattr(m, k)})
        return result

    def predict(
        self,
        messages: List[ChatAgentMessage],
        context: Optional[ChatContext] = None,  # noqa: ARG002
        custom_inputs: Optional[Dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        _ = context
        # Enforce thread_id presence for consistent checkpointing
        if not custom_inputs or not custom_inputs.get("thread_id"):
            raise ValueError("thread_id must be provided in custom_inputs")
        request = {"messages": self._convert_messages_to_dict(messages)}

        # Thread id for checkpointing/continuations
        thread_id = custom_inputs.get("thread_id")  # type: ignore[union-attr]
        thread_config = {"configurable": {"thread_id": thread_id}}

        # Load previous state strictly (Postgres-backed when configured)
        if not hasattr(self.app, "get_state"):
            raise RuntimeError("App does not support get_state; cannot resume with thread_id")
        state_values: Dict[str, Any] = self.app.get_state(thread_config).values  # type: ignore[attr-defined]

        # Initialize or update state
        if not state_values:
            # Build fresh state matching compile.py AgentState
            user_content = request["messages"][-1]["content"] if request["messages"] else ""
            to_send = {
                "original_prompt": user_content,
                "user_confirmed_hierarchy": None,
                "extracted_material": None,
                "extracted_location": None,
                "user_confirmed_location_hierarchy": None,
                "combined_prompt": None,
                "final_response": None,
                "messages": [request["messages"][-1]] if request["messages"] else [],
                "worker_outputs": {},
                "next_node": None,
            }
        else:
            to_send = state_values

        # # If the workflow signaled readiness for a new conversation, reset state
        # if isinstance(to_send, dict) and to_send.get("ready_for_new_conversation"):
        #     user_content = request["messages"][-1]["content"] if request["messages"] else ""
        #     to_send = {
        #         "original_prompt": user_content,
        #         "user_confirmed_hierarchy": None,
        #         "extracted_material": None,
        #         "extracted_location": None,
        #         "user_confirmed_location_hierarchy": None,
        #         "combined_prompt": None,
        #         "final_response": None,
        #         "messages": [request["messages"][-1]] if request["messages"] else [],
        #         "worker_outputs": {},
        #         "next_node": None,
        #     }

        # Optional inline user confirmations
        if check_if_heirarchy_resolver_exists(request):
            trigger_text = find_trigger_message(request, "Please confirm the material hierarchy level")
            updated_state = {
                "extracted_material": extract_material(trigger_text) if trigger_text else None,
                "user_confirmed_hierarchy": request["messages"][-1]["content"],
            }
            if hasattr(self.app, "update_state"):
                try:
                    self.app.update_state(thread_config, updated_state, "material_hierarchy_resolver_agent")  # type: ignore[attr-defined]
                    to_send = self.app.get_state(thread_config).values  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass

        elif check_if_location_heirarchy_resolver_exists(request):
            trigger_text = find_trigger_message(request, "Please confirm the location hierarchy level")
            updated_state = {
                "extracted_location": extract_location(trigger_text) if trigger_text else None,
                "user_confirmed_location_hierarchy": request["messages"][-1]["content"],
            }
            if hasattr(self.app, "update_state"):
                try:
                    self.app.update_state(thread_config, updated_state, "location_hierarchy_resolver_agent")  # type: ignore[attr-defined]
                    to_send = self.app.get_state(thread_config).values  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
        else:
            # Neither resolver flow is active; append the latest user message to original_prompt
            last_content = request["messages"][-1]["content"] if request.get("messages") else ""
            if last_content:
                original_prompt = (to_send.get("original_prompt") or "") if isinstance(to_send, dict) else ""
                if last_content not in original_prompt:
                    to_send["original_prompt"] = f"{original_prompt}\n{last_content}" if original_prompt else last_content
                # Also append the latest user message to the messages list with de-duplication
                if isinstance(to_send, dict):
                    base_messages = to_send.get("messages", [])
                    last_msg_dict = request["messages"][-1] if request.get("messages") else None
                    if last_msg_dict:
                        to_send["messages"] = add_message_if_not_exists(base_messages, last_msg_dict)
                # Persist these updates into the workflow state so the checkpointer captures them
                if hasattr(self.app, "update_state"):
                    try:
                        self.app.update_state(
                            thread_config,
                            {"original_prompt": to_send.get("original_prompt", ""), "messages": to_send.get("messages", [])},
                            "supervisor_agent",
                        )  # type: ignore[attr-defined]
                        to_send = self.app.get_state(thread_config).values  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass

        # Stream execution and capture final update; also capture interrupt messages (if any)
        last_chunk: Dict[str, Any] = {}
        interrupt_messages: List[Dict[str, Any]] = []
        for chunk in self.app.stream(to_send, config=thread_config, stream_mode="updates"):
            for node_id, value in chunk.items():
                if node_id == "__interrupt__":
                    try:
                        interrupt_obj = value[0] if isinstance(value, (list, tuple)) and value else value
                        payload = getattr(interrupt_obj, "value", None)
                        if isinstance(payload, dict):
                            raw_msgs = payload.get("messages", [])
                            extracted: List[Dict[str, Any]] = []
                            for m in raw_msgs:
                                if hasattr(m, "content") and hasattr(m, "type"):
                                    role = "user" if getattr(m, "type", "human").lower() == "human" else "assistant"
                                    extracted.append({"role": role, "content": getattr(m, "content", "")})
                                elif isinstance(m, dict) and "content" in m:
                                    extracted.append({"role": m.get("role", "assistant"), "content": m["content"]})
                            if extracted:
                                interrupt_messages = extracted
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    last_chunk = value

        # Prepare response messages
        final_messages = last_chunk.get("messages", []) if isinstance(last_chunk, dict) else []
        if not final_messages and interrupt_messages:
            final_messages = interrupt_messages

        def _to_role_content(m: Any) -> Dict[str, str]:
            if isinstance(m, dict):
                role = m.get("role", "assistant")
                content = m.get("content", "")
                return {"role": role, "content": content}
            # LangChain message objects
            content = getattr(m, "content", "")
            inferred_type = str(getattr(m, "type", "assistant")).lower()
            role = "user" if inferred_type in {"human", "humanmessage"} else "assistant"
            return {"role": role, "content": content}

        ms = [
            ChatAgentMessage(
                role=_to_role_content(msg)["role"],
                content=_to_role_content(msg)["content"],
                id=str(uuid.uuid4()),
            )
            for msg in final_messages
        ]

        custom_outputs = {"state": last_chunk, "thread_id": thread_id}
        return ChatAgentResponse(messages=ms, custom_outputs=custom_outputs)


mlflowapp = LangGraphChatAgent()
mlflow.models.set_model(mlflowapp)

