import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import sys

async def run_client():
    # Define server parameters to run main.py
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["main.py"],
        env={**os.environ, "SUPERSET_BASE_URL": "http://localhost:8088"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # Authenticate
            print("Authenticating...")
            auth_result = await session.call_tool("superset_auth_authenticate_user", arguments={"username": "admin", "password": "admin"})
            print(f"Auth Result: {auth_result.content[0].text}")
            print("Authenticated.")

            # Connect to Trino
            trino_ip = "192.168.68.118"
            print(f"Connecting to Trino at {trino_ip}...")
            try:
                result = await session.call_tool("superset_database_create", arguments={
                    "database_name": "Trino",
                    "sqlalchemy_uri": f"trino://admin@{trino_ip}:8080/iceberg",
                    "engine": "trino",
                    "configuration_method": "sqlalchemy_form"
                })
                print(f"Create Database Result: {result.content[0].text}")
            except Exception as e:
                print(f"Database connection error: {e}")

            # Call Agent
            print("\n--- Direct Database List Check ---")
            try:
                dbs = await session.call_tool("superset_database_list", arguments={})
                print(f"Databases found: {dbs.content[0].text}")
                
                # Check Schemas
                print("\n--- Direct Schema List Check ---")
                schemas = await session.call_tool("superset_database_schemas", arguments={"database_id": 1})
                print(f"Schemas found: {schemas.content[0].text}")
                
            except Exception as e:
                print(f"Error listing databases/schemas: {e}")

            query = "get the all the tables from the trino ip address ok list all the file it has iceberg.analytices"
            print(f"\nSending Query: {query}")
            
            try:
                result = await session.call_tool("superset_ai_analyze", arguments={"query": query})
                print("\nResult:")
                print(result.content[0].text)
            except Exception as e:
                print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(run_client())
