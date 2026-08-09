import json

notebook_content = {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "setup_cell",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from dotenv import load_dotenv\n",
    "from langchain_groq import ChatGroq\n",
    "from langchain_community.tools.wikipedia.tool import WikipediaQueryRun\n",
    "from langchain_community.utilities import WikipediaAPIWrapper\n",
    "from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper\n",
    "from langchain.agents import AgentExecutor, create_tool_calling_agent\n",
    "from langchain.tools import Tool\n",
    "from langchain_core.prompts import ChatPromptTemplate\n",
    "\n",
    "load_dotenv()\n",
    "\n",
    "llm_groq = ChatGroq(model=\"llama-3.1-8b-instant\", temperature=0, api_key=os.getenv(\"GROQ_API_KEY\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "tools_cell",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set up Wikipedia Tool\n",
    "wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())\n",
    "\n",
    "# Set up OpenWeatherMap Tool\n",
    "weather = OpenWeatherMapAPIWrapper()\n",
    "weather_tool = Tool(\n",
    "    name=\"Weather\",\n",
    "    func=weather.run,\n",
    "    description=\"Useful for when you need to answer questions about the current weather in a specific city.\"\n",
    ")\n",
    "\n",
    "# Combine tools\n",
    "tools = [wikipedia, weather_tool]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "agent_cell",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create prompt for the agent\n",
    "prompt = ChatPromptTemplate.from_messages([\n",
    "    (\"system\", \"You are a helpful assistant. Use the provided tools to answer questions.\"),\n",
    "    (\"human\", \"{input}\"),\n",
    "    (\"placeholder\", \"{agent_scratchpad}\"),\n",
    "])\n",
    "\n",
    "# Construct the Tools agent\n",
    "agent = create_tool_calling_agent(llm_groq, tools, prompt)\n",
    "\n",
    "# Create an agent executor by passing in the agent and tools\n",
    "agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "run_weather",
   "metadata": {},
   "outputs": [],
   "source": [
    "response1 = agent_executor.invoke({\"input\": \"What is the current weather in Chennai?\"})\n",
    "print(response1['output'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "run_wiki",
   "metadata": {},
   "outputs": [],
   "source": [
    "response2 = agent_executor.invoke({\"input\": \"Who created the Python programming language?\"})\n",
    "print(response2['output'])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('d:/LLM/Chapter 2/tools.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)
