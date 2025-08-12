import uuid
import os
from functools import partial
from langgraph.types import Command
from dotenv import load_dotenv
from contextlib import contextmanager
from psycopg import connect
from psycopg.rows import dict_row

import mlflow

# LangGraph imports
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

# Databricks-specific imports
from databricks_langchain.genie import GenieAgent
from databricks_langchain import ChatDatabricks

# Local imports (package-qualified)
from agent_build.utils import setup_logging
from agent_build.agents import (
    AgentState,
    material_hierarchy_resolver_agent as _material_agent,
    location_hierarchy_resolver_agent as _location_agent,
    p2p_spend_genie_agent as _genie_agent,
    summary_agent as _summary_agent,
    supervisor_agent as _supervisor_agent,
)
from agent_build.config import default_config
from agent_build.utils import build_db_uri

# Enable MLflow autologging
mlflow.langchain.autolog()

# Load environment variables
load_dotenv(".env")


# =============================================================================
# CONFIGURATION AND INITIALIZATION
# =============================================================================

# Configure logging level here (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = "WARNING"

# Initialize logger
logger = setup_logging(LOG_LEVEL)

# Initialize LLM
llm = ChatDatabricks(endpoint="databricks-claude-3-7-sonnet")

# =============================================================================
# DATABASE AND CHECKPOINTER SETUP
# =============================================================================

# Database configuration
username = os.getenv("CLIENT_ID")
instance_name = os.getenv("DB_INSTANCE_NAME")
databricks_host = os.getenv("DATABRICKS_HOST")


print("instance_name:", instance_name)

# Initialize store/checkpointer lazily within get_pg_checkpointer()

@contextmanager
def get_pg_checkpointer():
    """Yield a PostgresSaver backed by a psycopg connection with robust settings.

    Ensures autocommit and dict_row, and adds TCP keepalive parameters to reduce
    unexpected SSL socket closures on long-running streams.
    """
    if not (username and instance_name):
        yield None
        return


    uri = build_db_uri(username, instance_name, use_sp=True)
    print("uri:", uri)

    # Append keepalive parameters to URI to reduce idle disconnects
    keepalive_params = "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
    if "?" in uri:
        if "keepalives=" not in uri:
            uri = f"{uri}&{keepalive_params}"
    else:
        uri = f"{uri}?{keepalive_params}"

    # IMPORTANT: Keep the connection open for the duration of the caller using this saver.
    # Do NOT close the connection on context exit. The caller is responsible for lifecycle.
    conn = connect(uri, autocommit=True, row_factory=dict_row)
    saver = PostgresSaver(conn)
    try:
        yield saver
    finally:
        # Intentionally do not close the connection here to avoid breaking long-lived streams/state loads.
        # The process exit or explicit shutdown should close it.
        pass

# Genie agent configuration
P2P_SPEND_GENIE_SPACE_ID = "01f0421695fe1fb694762b30f68f799d"
p2p_spend_genie = GenieAgent(P2P_SPEND_GENIE_SPACE_ID, "p2p_spend_genie")

"""Agents are imported from `agent_build/agents.py`; this file wires them into a graph."""

# =============================================================================
# WORKFLOW DEFINITION
# =============================================================================

def start_node(state: AgentState) -> AgentState:
    """Initialize the workflow."""
    logger.info("🚀 Workflow started")
    logger.debug("Initial state: %s", state)
    return state

# Create workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("start", start_node)
workflow.add_node(
    "material_hierarchy_resolver_agent",
    partial(_material_agent, llm_model=llm, logger=logger, default_config=default_config, utils=__import__("agent_build.utils", fromlist=["*"]))
)
workflow.add_node(
    "location_hierarchy_resolver_agent",
    partial(_location_agent, llm_model=llm, logger=logger, default_config=default_config, utils=__import__("agent_build.utils", fromlist=["*"]))
)
workflow.add_node(
    "summary_agent",
    partial(_summary_agent, llm_model=llm, default_config=default_config, logger=logger)
)
workflow.add_node(
    "p2p_spend_genie_agent",
    partial(_genie_agent, genie_agent=p2p_spend_genie, default_config=default_config, utils=__import__("agent_build.utils", fromlist=["*"]), logger=logger)
)
workflow.add_node("supervisor_agent", partial(_supervisor_agent, llm_model=llm, logger=logger, default_config=default_config))

# Define entry point
workflow.set_entry_point("start")

# Define transitions
workflow.add_edge("start", "supervisor_agent")

# Conditional edges based on supervisor decisions
workflow.add_conditional_edges(
    "supervisor_agent",
    lambda state: state["next_node"],
    {
        "material_hierarchy_resolver_agent": "material_hierarchy_resolver_agent",
        "location_hierarchy_resolver_agent": "location_hierarchy_resolver_agent",
        "p2p_spend_genie_agent": "p2p_spend_genie_agent",
        "summary_agent": "summary_agent"
    }
)

# Return to supervisor after each agent
workflow.add_edge("material_hierarchy_resolver_agent", "supervisor_agent")
workflow.add_edge("location_hierarchy_resolver_agent", "supervisor_agent")
workflow.add_edge("p2p_spend_genie_agent", "supervisor_agent")

# Compile workflow is done at runtime in main()


# =============================================================================
# EXECUTION AND STREAM PROCESSING
# =============================================================================

def process_stream(app, initial_state, thread_config):
    """
    Process a LangGraph stream, handling human-in-the-loop interruptions by
    resuming the graph with a Command(resume=...) payload.
    """
    try:
        to_send = initial_state
        while True:
            interrupted = False
            for chunk in app.stream(to_send, config=thread_config, stream_mode='updates'):
                for node_id, value in chunk.items():
                    # print("node_id:", node_id)
                    # print("value:", value)

                    if node_id == "__interrupt__":
                        logger.info("⏸️  Human-in-the-loop interruption")
                        user_feedback = input("Please provide the hierarchy level: ").strip()
                        # Resume execution with provided feedback
                        to_send = Command(resume=user_feedback)
                        interrupted = True
                        break

                    if node_id == "__end__":
                        logger.info("🏁 Workflow complete")
                        return

                    if node_id == "summary_agent":
                        logger.info("📋 Final summary generated")
                        return

                if interrupted:
                    break

            if not interrupted:
                # No interruptions encountered; streaming finished
                return
    except Exception as e:
        logger.error("Error in process_stream: %s", e)
        raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    def main():  
        try:
            # Compile workflow with context-managed checkpointer when available
            with get_pg_checkpointer() as checkpointer:
                if checkpointer is not None:
                    checkpointer.setup()
                    app = workflow.compile(checkpointer=checkpointer)
                    print("✅ Workflow compiled with PostgreSQL checkpointer")
                else:
                    app = workflow.compile()
                    print("⚠️  Workflow compiled with in-memory checkpointer (no persistence)")
            
                # Single run example
                prompt1 = "What is the total spend of tea in India in 2024?"
                thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                initial_state = {
                    "original_prompt": prompt1,
                    "user_confirmed_hierarchy": None,
                    "extracted_material": None,
                    "extracted_location": None,
                    "user_confirmed_location_hierarchy": None,
                    "combined_prompt": None,
                    "final_response": None,
                    "messages": [],
                    "worker_outputs": {},
                    "next_node": None,
                }
                print(f"🚀 Starting workflow with prompt: {prompt1}")
                print(f"📝 Thread ID: {thread_config['configurable']['thread_id']}")
                process_stream(app, initial_state, thread_config)
        
        except Exception as e:
            logger.error("❌ Error in main: %s", e)
            raise

    # Run the async main function
    main()
