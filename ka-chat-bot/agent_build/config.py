"""
Configuration file for procurement agent system.
Contains dataclasses for prompts, messages, and system parameters.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class MaterialPrompt:
    """Configuration for material extraction prompts."""
    
    def build_prompt(self, user_prompt: str) -> str:
        """Build the material extraction prompt."""
        return f"""
        You are a helpful assistant working with procurement data for a CPG company.

        The material data is tagged across 5 levels of hierarchy:
        - category
        - sub_category
        - product_line
        - sku_group
        - material_name

        Some example materials include:
        tea, coffee, fruits and vegetables, packaging materials (e.g., plastic film roll, PET bottle, corrugated box), dairy products, sauces, etc.

        ---
        Your task is to read a user prompt and do two things:

        1. Extract the material name being referred to — usually a noun like "coffee" or "packaging".
        2. Extract the **explicitly mentioned hierarchy level** if one is **clearly stated** — otherwise, return `None`.

        3. Do not guess the hierarchy level. Do not default to "material_name" if it is not mentioned.
        4. Only return a hierarchy level like "material_name" or "category" if it is explicitly stated (e.g., "at the SKU group level" or "by category").
        5. If the user says things like "group by" or "filter by", ignore it — that doesn't count as explicitly specifying the hierarchy.
        6. If the prompt doesn't clearly say a level, return `None`.

        ---
        Format your output exactly like this (no extra text):
        Material: <material or None>  
        Hierarchy: <one of: category, sub_category, product_line, sku_group, material_name, or None>

        ---
        Prompt:
        "{user_prompt}"
        """


@dataclass
class LocationPrompt:
    """Configuration for location extraction prompts."""
    
    def build_prompt(self, user_prompt: str) -> str:
        """Build the location extraction prompt."""
        return f"""
        You are a helpful assistant working with procurement data for a CPG company. The data contains location hierarchies.

        Each location follows a 3-level hierarchy:
        - region_name
        - cluster_name
        - country_name

        Your job is to extract:
        1. The **location name** mentioned in the prompt — this is typically a geographic noun (e.g., "India", "EMEA", "AMER", "APAC", "China", "Northern Europe" etc)
        2. The **hierarchy level** ONLY IF it is *explicitly stated* using keywords like:
        - "at the country level"
        - "for the region"
        - "by cluster"

        Do **NOT infer or assume** the hierarchy based on the location.

        Format:
        location: <location name or None>  
        hierarchy: <region_name, cluster_name, country_name, or None>

        User Prompt:
        \"{user_prompt}\"
        """


@dataclass
class SummaryPrompt:
    """Configuration for summary generation prompt."""

    def build_prompt(self, messages: List[Any]) -> str:
        return f"""
        You are a Procurement Insights Assistant. Your task is to summarize Genie agent's response in a clear, friendly, and actionable format for a business user.

        Content to summarize:
        "{messages}"

        Instructions:
        - Focus only on the response from the Genie agent.
        - Use the exact number format returned (do NOT convert to millions, use scientific notation, or round off values).
        - If Genie provided valid data (like total spend), summarize it using:
            • Plain bullet points or short paragraphs or in tabular format, as applicable.
        - Exact figures (e.g., $693,986,110.05)
        - The spend amount returned by the Genie agent is always represented in Euros (€). Never assume or convert the currency to any other unit.
        - If Genie returned no results, null, error, or blank values:
            • Mention that no meaningful data was returned
            • Refer to earlier user messages like "no matching material" for context
            • Suggest the user refine or rephrase the question
        - Make the output sound helpful and business-friendly
        - Do NOT include technical phrases like "summary response" or agent names
        - You can optionally offer the user to ask for further breakdowns (e.g., by region, supplier)

        Be accurate, clear, and conversational.
        """


@dataclass
class SupervisorPrompt:
    """Configuration for supervisor agent decision prompt."""

    def build_prompt(self, state: Dict[str, Any], messages: List[Any]) -> str:
        return (
            "You are a supervisor coordinating a multi-step workflow for answering procurement questions.\n\n"
            f"Here is the current context:\n- Current State: {state}\n- Messages so far: {messages}\n\n"
            "You must choose the **next agent to call**, strictly following these rules:\n"
            "1. Each agent must be called **only once** per workflow run.\n"
            "2. Agents already present in 'worker_outputs' have been called. DO NOT call them again.\n"
            "3. Proceed in the following order:\n"
            "- First: material_hierarchy_resolver_agent\n"
            "- Then: location_hierarchy_resolver_agent\n"
            "- Then: p2p_spend_genie_agent\n"
            "- Finally: summary_agent\n"
            "4. If all previous steps are done, call 'summary_agent' to complete the workflow.\n\n"
            "Choose exactly one of: 'material_hierarchy_resolver_agent', 'location_hierarchy_resolver_agent', 'p2p_spend_genie_agent', 'summary_agent'\n\n"
            "Next agent to call:"
        )


@dataclass
class SystemMessages:
    """Configuration for system messages."""
    
    # Material-related messages
    material_skipped: str = "User skipped material hierarchy confirmation."
    material_confirmed: str = "Confirmed hierarchy is '{hierarchy}' for the material '{material}'."
    material_no_hierarchy: str = "User indicated no specific hierarchy even if there is hierarchy level ambiguity identified"
    
    # Location-related messages
    location_skipped: str = "User skipped location hierarchy confirmation."
    location_confirmed: str = "Confirmed hierarchy is '{hierarchy}' for location '{location}'."
    location_no_hierarchy: str = "No location or hierarchy found."
    
    # Error messages
    genie_error: str = "Error: Genie called without combined_prompt."
    genie_input_validation: str = "Error: {validation_error}"
    genie_processing_error: str = "Error processing Genie response: {error}"
    genie_timeout: str = "Genie agent request timed out after {timeout} seconds."
    genie_empty_response: str = "Genie agent returned empty or invalid response."
    llm_unavailable: str = "LLM unavailable for extraction."
    
    # Success messages
    extraction_success: str = "Extracted {item_type}: '{item}' and hierarchy: '{hierarchy}'."
    hierarchy_resolved: str = "Resolved hierarchy from lookup: '{hierarchy}'."
    
    def format_message(self, template: str, **kwargs) -> str:
        """Format a message template with provided arguments."""
        return template.format(**kwargs)


@dataclass
class GenieConfig:
    """Configuration for Genie agent."""
    
    model_name: str = "databricks-dbrx-instruct"
    temperature: float = 0.1
    max_tokens: int = 4000
    top_p: float = 0.95
    top_k: int = 40
    
    # Function calling configuration
    function_calling: str = "auto"
    
    # Response formatting
    response_format: Dict[str, str] = None
    
    # Genie-specific settings
    min_prompt_length: int = 10
    max_prompt_length: int = 10000
    timeout_seconds: int = 300
    retry_attempts: int = 3
    
    # Response validation
    require_content: bool = True
    validate_response_format: bool = True
    
    def __post_init__(self):
        """Set default response format if not provided."""
        if self.response_format is None:
            self.response_format = {"type": "text"}


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    level: str = "INFO"
    format_debug: str = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-25s | %(message)s'
    format_info: str = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
    date_format: str = '%H:%M:%S'
    logger_name: str = "procurement_agent"


@dataclass
class UCConfig:
    """Configuration for Unity Catalog functions."""
    
    material_function: str = "malay_demo.procurement.get_material_hierarchy_level"
    location_function: str = "malay_demo.procurement.get_location_hierarchy_level"
    
    # Function parameters
    material_param: str = "material_name"
    location_param: str = "location_name"
    
    # Error handling
    default_error_response: str = "no hierarchy identified"
    ambiguous_response: str = "multiple"


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    
    # User interaction settings
    max_retries: int = 3
    skip_keywords: List[str] = None
    confirmation_timeout: int = 300  # seconds
    
    # Extraction settings
    require_explicit_hierarchy: bool = True
    allow_hierarchy_inference: bool = False
    
    def __post_init__(self):
        """Set default skip keywords if not provided."""
        if self.skip_keywords is None:
            self.skip_keywords = ["skip", "none", "no"]


@dataclass
class DefaultConfig:
    """Main configuration class that combines all configurations."""
    
    # Component configurations
    material_prompt: MaterialPrompt = None
    location_prompt: LocationPrompt = None
    messages: SystemMessages = None
    genie: GenieConfig = None
    logging: LoggingConfig = None
    uc: UCConfig = None
    agent: AgentConfig = None
    summary_prompt: 'SummaryPrompt' = None
    supervisor_prompt: 'SupervisorPrompt' = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.material_prompt is None:
            self.material_prompt = MaterialPrompt()
        if self.location_prompt is None:
            self.location_prompt = LocationPrompt()
        if self.messages is None:
            self.messages = SystemMessages()
        if self.genie is None:
            self.genie = GenieConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.uc is None:
            self.uc = UCConfig()
        if self.agent is None:
            self.agent = AgentConfig()
        # Initialize prompt templates
        if self.summary_prompt is None:
            # Late import to avoid forward reference issues
            self.summary_prompt = SummaryPrompt()
        if self.supervisor_prompt is None:
            self.supervisor_prompt = SupervisorPrompt()


# Create default configuration instance
default_config = DefaultConfig() 