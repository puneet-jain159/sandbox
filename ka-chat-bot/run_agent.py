import uuid
from typing import Optional, Dict, Any

from agent_build.mlflowlogger import LangGraphChatAgent


def create_input_example(
    query: str = "What is the total spend of tea in 2023?",
    thread_id: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create an example input for the model.

    Matches ChatAgentRequest schema exactly.
    """
    example = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ],
        "custom_inputs": {
            "thread_id": thread_id or "1234",
        },
    }

    if state is not None:
        example["custom_inputs"]["state"] = state

    return example


def _coerce_messages_to_dict(messages: list[dict] | list[Any]) -> list[dict]:
    coerced: list[dict] = []
    for m in messages:
        if isinstance(m, dict):
            if "role" in m and "content" in m:
                coerced.append({"role": m["role"], "content": m["content"]})
        else:
            role = getattr(m, "role", "assistant")
            content = getattr(m, "content", "")
            coerced.append({"role": role, "content": content})
    return coerced


def continue_conversation(model: LangGraphChatAgent, response, query: str, thread_id: str | None = None):
    """
    Continue the conversation with the model.
    """
    prev_msgs = _coerce_messages_to_dict(getattr(response, "messages", []))
    prev_msgs.append({"role": "user", "content": query})

    return model.predict(
        messages=prev_msgs,
        custom_inputs={"thread_id": str(thread_id or "1234")},
    )


def main() -> None:
    # Build agent (auto-loads workflow from agent_build.compile)
    agent = LangGraphChatAgent()

    # Prepare example request with a unique thread_id
    thread_id = str(uuid.uuid4())
    example = create_input_example(query="What is the total spend of tea in 2023?", thread_id=thread_id)

    # Call predict using the example
    response = agent.predict(
        messages=example["messages"],
        custom_inputs=example["custom_inputs"],
    )

    # Print results
    print("Initial response messages:")
    for m in _coerce_messages_to_dict(response.messages):
        print(f"- {m['role']}: {m['content']}")

    # print("\nCustom outputs:")
    # print(response.custom_outputs)

    # Continue conversation: specify hierarchy
    response = continue_conversation(
        model=agent,
        response=response,
        query="skip",
        thread_id=thread_id,
    )
    print("\nAfter hierarchy response messages:")
    for m in _coerce_messages_to_dict(response.messages):
        print(f"- {m['role']}: {m['content']}")

    # # Continue conversation: specify location hierarchy
    # response = continue_conversation(
    #     model=agent,
    #     response=response,
    #     query="country_name",
    #     thread_id=thread_id,
    # )
    # print("\nAfter location hierarchy response messages:")
    # for m in _coerce_messages_to_dict(response.messages):
    #     print(f"- {m['role']}: {m['content']}")

    
    response = continue_conversation(
        model=agent,
        response=response,
        query="What is the total spend of coffee in 2023?",
        thread_id=thread_id,
    )
    print("\nAfter location hierarchy response messages:")
    for m in _coerce_messages_to_dict(response.messages):
        print(f"- {m['role']}: {m['content']}")


if __name__ == "__main__":
    main()

