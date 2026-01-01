import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import ollama
from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)

# System prompt for the Superset Agent
SYSTEM_PROMPT = """You are a Superset Data Analyst Agent. Your goal is to help users visualize data.
You have access to a Superset instance with Trino/Database connections.

Follow this process:
1. Understand the User's Request.
2. INTERSPECT Data: Use `superset_database_list` to find databases.
   - For Trino/Presto, ALWAYS check schemas using `superset_database_schemas(database_id)` first.
   - Then use `superset_database_get_tables(database_id, schema_name=...)` with the relevant schema.
3. QUERY Data (Optional but recommended): Use `superset_sqllab_execute_query` to preview data or verify assumptions.
4. VISUALIZE: Use `superset_chart_create` to create a chart.
   - Choose an appropriate `viz_type` (e.g., 'table', 'big_number', 'echarts_timeseries_line', 'echarts_timeseries_bar', 'pie').
   - `datasource_type` should be 'table' (if using a dataset) or 'query' (if creating from SQL).
   - Construct the `params` JSON carefully based on the chart type.
5. FINAL ANSWER: Return the URL of the created chart or dashboard.

Tools available:
- `superset_database_list`: List available databases.
- `superset_database_get_tables(database_id)`: List tables in a database.
- `superset_sqllab_execute_query(database_id, sql)`: Execute SQL.
- `superset_chart_create(slice_name, datasource_id, datasource_type, viz_type, params)`: Create a chart.
- `superset_dataset_list()`: List available datasets.

Response Format:
You must STRICTLY use the provided tools.
If you need more information, ask the user.
"""

class SupersetAgent:
    def __init__(self, ctx: Context, model: str = "llama3"):
        self.ctx = ctx
        self.model = model
        self.client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def chat(self, query: str) -> str:
        """
        Process a user query through the Ollama agent loop.
        """
        self.history.append({"role": "user", "content": query})
        
        # Available tools definition for Ollama
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "superset_database_list",
                    "description": "List all databases available in Superset",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "superset_database_get_tables",
                    "description": "List tables in a specific database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "database_id": {"type": "integer", "description": "The ID of the database"},
                            "schema_name": {"type": "string", "description": "The schema name (optional but recommended for Trino)"}
                        },
                        "required": ["database_id"],
                    },
                },
            },
             {
                "type": "function",
                "function": {
                    "name": "superset_database_schemas",
                    "description": "List schemas in a specific database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "database_id": {"type": "integer", "description": "The ID of the database"}
                        },
                        "required": ["database_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "superset_sqllab_execute_query",
                    "description": "Execute a SQL query against a database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "database_id": {"type": "integer", "description": "The ID of the database"},
                            "sql": {"type": "string", "description": "The SQL query to execute"}
                        },
                        "required": ["database_id", "sql"],
                    },
                },
            },
             {
                "type": "function",
                "function": {
                    "name": "superset_chart_create",
                    "description": "Create a new chart in Superset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slice_name": {"type": "string", "description": "Name of the chart"},
                            "datasource_id": {"type": "integer", "description": "ID of the datasource (table or saved query)"},
                            "datasource_type": {"type": "string", "enum": ["table", "query"], "description": "Type of datasource"},
                            "viz_type": {"type": "string", "description": "Type of visualization (e.g., echarts_timeseries_bar)"},
                            "params": {"type": "object", "description": "Chart configuration parameters"}
                        },
                        "required": ["slice_name", "datasource_id", "datasource_type", "viz_type", "params"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "superset_dataset_list",
                    "description": "List available datasets",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        logger.info(f"Processing query: {query}")

        # Main Agent Loop
        for _ in range(5): # Max 5 turns to prevent infinite loops
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=self.history,
                    tools=tools,
                )
            except Exception as e:
                return f"Error communicating with Ollama: {str(e)}"

            message = response['message']
            self.history.append(message)

            tool_calls = message.get('tool_calls')
            if not tool_calls:
                # No tool call, just a text response implies we are done or asking a question
                return message['content']

            # Process tool calls
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']
                
                logger.info(f"Agent calling tool: {function_name} with args: {arguments}")
                
                tool_result = await self._execute_tool(function_name, arguments)
                
                # Add tool output to history
                self.history.append({
                    "role": "tool",
                    "content": json.dumps(tool_result, default=str),
                    "name": function_name,
                })

        return "Agent loop limit reached without final answer."

    async def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        # Import tools dynamically to avoid circular imports or access global 'mcp' context methods if possible
        # Since we are inside the MCP server process, we can access the functions directly if we import them
        # However, the functions in main.py use @mcp.tool decorator which might wrap them.
        # We need to call the underlying implementation or invoke them via the stored tool map if available.
        # For simplicity, we will assume we can import the handlers from main.py or similar.
        
        # WARNING: Circular import if we import main here. 
        # Strategy: Pass the tool functions or use a dispatch mechanism.
        # Ideally, main.py should pass the tool callables to Agent, or Agent calls them via a registry.
        
        # Alternative: We will move the tool implementations to a separate file 'tools.py' later or 
        # just import them inside the method to avoid top-level circular dependency.
        from main import (
            superset_database_list,
            superset_database_get_tables,
            superset_database_schemas,
            superset_sqllab_execute_query,
            superset_chart_create,
            superset_dataset_list
        )

        try:
            if name == "superset_database_list":
                return await superset_database_list(self.ctx)
            elif name == "superset_database_schemas":
                return await superset_database_schemas(self.ctx, **args)
            elif name == "superset_database_get_tables":
                return await superset_database_get_tables(self.ctx, **args)
            elif name == "superset_sqllab_execute_query":
                return await superset_sqllab_execute_query(self.ctx, **args)
            elif name == "superset_chart_create":
                return await superset_chart_create(self.ctx, **args)
            elif name == "superset_dataset_list":
                return await superset_dataset_list(self.ctx)
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return {"error": str(e)}
