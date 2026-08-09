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
    "from langgraph.prebuilt import create_react_agent\n",
    "from langchain.tools import Tool\n",
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
    "# Create the ReAct agent using LangGraph\n",
    "agent = create_react_agent(llm_groq, tools=tools)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "run_weather",
   "metadata": {},
   "outputs": [],
   "source": [
    "response1 = agent.invoke({\"messages\": [(\"user\", \"What is the current weather in Chennai?\")]})\n",
    "print(response1['messages'][-1].content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "run_wiki",
   "metadata": {},
   "outputs": [],
   "source": [
    "response2 = agent.invoke({\"messages\": [(\"user\", \"Who created the Python programming language?\")]})\n",
    "print(response2['messages'][-1].content)"
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
