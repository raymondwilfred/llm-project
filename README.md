# 🧠 Second Brain AI — LangGraph Agent & Personal Knowledge Assistant

A Streamlit application that combines a **LangGraph ReAct agent**, **Groq's Llama 3.3 70B**, and a set of live tools (Wikipedia, weather, date/time, calculator) into a personal "second brain" — an AI-powered knowledge and productivity assistant with persistent memory, smart notes, task management, research tools, flashcards, a daily journal, and a focus timer.

The repo also includes a lightweight standalone chat demo and a few notebook-generation utility scripts used during development.

---

## ✨ Features

**`second_brain.py`** — the main app:

- 🤖 **AI Chat Agent** — LangGraph ReAct agent (Llama 3.3 70B via Groq) with persistent per-session memory and 4 tools: Wikipedia search, live weather (OpenWeatherMap), current date/time, and a safe calculator
- 📝 **Smart Notes** — capture notes, tag them, search, and AI-enhance or AI-summarize any note
- ✅ **Task Manager** — priority-based task tracking with AI-suggested tasks
- 🔬 **Research Assistant** — topic research (Quick/Standard/Deep/Expert depth), text summarization, concept comparison, and expert Q&A
- 💡 **Idea Generator** — brainstorming, startup ideas, content ideas, and structured problem-solving (First Principles, SCAMPER, Design Thinking, Six Thinking Hats)
- 🌤️ **Weather & Info** — live weather lookup and Wikipedia search
- 📚 **Flashcards** — AI-generated flashcard decks with a study mode
- 📓 **Daily Journal** — mood tracking, gratitude, highlights, and AI reflection
- ⏱️ **Focus Timer** — Pomodoro-style focus sessions

**`app.py`** — a minimal LangChain + Groq + Streamlit chat demo, useful as a quick reference for the core chain (prompt → LLM → parser) before diving into the full app.

**Other scripts** — `create_tools_nb*.py`, `fix.py`, `fix_dotenv.py`, `fix_wiki.py` are development utilities used to generate/patch the accompanying `demo.ipynb` notebook.

---

## 🏗️ Tech Stack

- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Orchestration:** LangChain + LangGraph (`create_react_agent`, `MemorySaver` checkpointing)
- **UI:** Streamlit
- **Tools:** Wikipedia API, OpenWeatherMap API, custom calculator/datetime tools
- **Package management:** [uv](https://github.com/astral-sh/uv) (`pyproject.toml`, `uv_build` backend)

---

## 📦 Requirements

- Python 3.12+
- A [Groq API key](https://console.groq.com/keys)
- An [OpenWeatherMap API key](https://openweathermap.org/api) (for weather features)

---

## 🚀 Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/raymondwilfred/llm-project.git
   cd llm-project
   ```

2. **Install dependencies** (using [uv](https://github.com/astral-sh/uv))
   ```bash
   uv sync
   ```
   or with pip:
   ```bash
   pip install -e .
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key
   OPENWEATHER_API_KEY=your_openweathermap_api_key
   ```

4. **Run the app**

   Full Second Brain app:
   ```bash
   streamlit run second_brain.py
   ```

   Minimal chat demo:
   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

```
llm-project/
├── second_brain.py        # Main app — full Second Brain assistant
├── app.py                 # Minimal LangChain + Groq + Streamlit chat demo
├── create_tools_nb.py      # Notebook generation utility (v1)
├── create_tools_nb_v2.py   # Notebook generation utility (v2)
├── create_tools_nb_v3.py   # Notebook generation utility (v3)
├── create_tools_nb_v4.py   # Notebook generation utility (v4)
├── demo.ipynb              # Generated demo notebook
├── fix.py                  # Notebook/dev patch script
├── fix_dotenv.py           # .env handling patch script
├── fix_wiki.py             # Wikipedia tool patch script
├── pyproject.toml           # Project metadata & dependencies (uv)
└── .python-version
```

---

## 📝 Notes

- Weather lookups require `OPENWEATHER_API_KEY` (or `OPENWEATHERMAP_API_KEY`) to be set; without it, weather queries will fail.
- Notes, tasks, journal entries, and flashcards are stored in Streamlit session state, so they reset when the app restarts. For persistence across sessions, you'd need to add a database or file-based store.

---

## 📄 License

No license specified yet — add one (e.g., MIT) if you intend for others to reuse this code.
