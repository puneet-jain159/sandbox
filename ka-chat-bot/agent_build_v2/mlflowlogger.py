from typing import Dict, Any, Optional, List, Generator
import uuid
import mlflow
from mlflow.pyfunc import ChatAgent  # type: ignore
from langchain_community.adapters.openai import convert_message_to_dict
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext ,ChatAgentChunk # type: ignore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
# Load environment variables early to ensure database connection works
from dotenv import load_dotenv
load_dotenv(".env")







# Using mlflow ChatAgent types to enable model logging and deployment


class LangGraphChatAgent(ChatAgent):
    """Adapter that wraps a LangGraph compiled app to the MLflow ChatAgent API.

    Expects an app with methods: stream(state, config, stream_mode), and optionally
    get_state(thread_config) and update_state(thread_config, updates, node_id).
    """

    def __init__(self, app_or_workflow: Optional[Any] = None):
        # Accept either a compiled app (with stream) or a StateGraph-like workflow (with compile)
        self.graph = None
        # If not provided, fetch workflow from compile.py
        if app_or_workflow is None:
            try:
                from agent_build_v2.run_toolnode import graph # lazy import
                from agent_build.compile import get_pg_checkpointer  # lazy import to avoid cycles
                app_or_workflow = graph
            except Exception as _e:  # noqa: F841
                raise RuntimeError(f"Failed to import workflow from agent_build_v2.run_toolnode: {_e}") from _e
        
        # Check if it's already a compiled app
        if hasattr(app_or_workflow, "stream"):
            self.graph = app_or_workflow
        else:
            # Compile with checkpointer strictly; require Postgres when available
            from agent_build.compile import get_pg_checkpointer  # lazy import to avoid cycles
            with get_pg_checkpointer() as checkpointer:  # type: ignore[misc]
                self.graph = app_or_workflow.compile(checkpointer=checkpointer)
        
        if self.graph is None:
            raise RuntimeError("Failed to initialize graph - self.graph is None")

    def _convert_messages_to_langchain(self, messages: List[ChatAgentMessage]) -> List[Any]:
        """Convert MLflow ChatAgentMessage objects to LangChain BaseMessage objects."""
        
        result = []
        for m in messages:
            if isinstance(m, dict):
                # Handle dict-style messages
                role = m.get("role", "user")
                content = m.get("content", "")
                
                if role == "user":
                    result.append(HumanMessage(content=content))
                elif role == "assistant":
                    if "tool_calls" in m and m["tool_calls"]:
                        # Handle assistant message with tool calls
                        tool_calls = []
                        for tc in m["tool_calls"]:
                            tool_calls.append({
                                "id": tc.get("id", str(uuid.uuid4())),
                                "name": tc["function"]["name"],
                                "args": tc["function"]["arguments"]
                            })
                        result.append(AIMessage(content=content, tool_calls=tool_calls))
                    else:
                        result.append(AIMessage(content=content))
                elif role == "system":
                    result.append(SystemMessage(content=content))
                elif role == "tool":
                    tool_call_id = m.get("tool_call_id")
                    name = m.get("name", "unknown_tool")
                    if tool_call_id:
                        result.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=name))
                    else:
                        # Skip tool messages without tool_call_id as they'll cause errors
                        continue
            else:
                # Handle ChatAgentMessage objects
                role = getattr(m, 'role', 'user')
                content = getattr(m, 'content', '')
                
                if role == "user":
                    result.append(HumanMessage(content=content))
                elif role == "assistant":
                    result.append(AIMessage(content=content))
                elif role == "system":
                    result.append(SystemMessage(content=content))
                elif role == "tool":
                    tool_call_id = getattr(m, 'tool_call_id', None)
                    name = getattr(m, 'name', 'unknown_tool')
                    if tool_call_id:
                        result.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=name))
                    else:
                        # Skip tool messages without tool_call_id
                        continue
        
        return result

    def _convert_langchain_message_to_dict(self, msg: Any) -> Dict[str, Any]:
        """Convert LangChain message objects to dictionary format for MLflow."""
        
        try:
            # Use LangChain's built-in converter
            result = convert_message_to_dict(msg)
            
            # Ensure the message has an ID for MLflow compatibility
            if "id" not in result:
                result["id"] = str(uuid.uuid4())
            
            return result
        except Exception:
            # Fallback for cases where convert_message_to_dict fails
            if isinstance(msg, dict):
                # If it's already a dict, ensure it has required fields
                if "id" not in msg:
                    msg["id"] = str(uuid.uuid4())
                return msg
            else:
                # Fallback for unknown message types
                return {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": str(msg)
                }

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_langchain(messages)}

        if not custom_inputs or not custom_inputs.get("thread_id"):
            raise ValueError("thread_id must be provided in custom_inputs")

        # Thread id for checkpointing/continuations
        thread_id = custom_inputs.get("thread_id")  # type: ignore[union-attr]
        thread_config = {"configurable": {"thread_id": thread_id}}

        messages = []
        for event in self.graph.stream(request, config=thread_config, stream_mode="updates"):
            for node_data in event.values():
                # Convert LangChain messages to MLflow format
                for msg in node_data.get("messages", []):
                    converted_msg = self._convert_langchain_message_to_dict(msg)
                    messages.append(ChatAgentMessage(**converted_msg))
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_langchain(messages)}

        if not custom_inputs or not custom_inputs.get("thread_id"):
            raise ValueError("thread_id must be provided in custom_inputs")

        # Thread id for checkpointing/continuations
        thread_id = custom_inputs.get("thread_id")  # type: ignore[union-attr]
        thread_config = {"configurable": {"thread_id": thread_id}}

        for event in self.graph.stream(request, config=thread_config, stream_mode="updates"):
            for node_data in event.values():
                for msg in node_data.get("messages", []):
                    converted_msg = self._convert_langchain_message_to_dict(msg)
                    # Create ChatAgentChunk with proper structure
                    # The delta should contain the message content in the correct format
                    chunk = ChatAgentChunk(
                        delta=converted_msg,
                        finish_reason=None,
                        usage=None
                    )
                    yield chunk



mlflowapp = LangGraphChatAgent()
mlflow.models.set_model(mlflowapp)

