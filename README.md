# 🤖 Voice-Powered Agentic AI Coding Assistant

A browser-free, voice-controlled AI assistant that listens to your programming requests, interprets them using LLMs, creates a coding plan, generates Python files, executes them, and reads the results aloud. Powered by **LangGraph**, **LangChain**, **OpenAI GPT-4o**, and **MongoDB**.

---



---
## Execution Flowchart
<img width="2987" height="975" alt="Cursor" src="https://github.com/user-attachments/assets/5050843e-1621-4049-85b0-5eadbd7f512d" />


## ✨ Features
- 🎤 **Voice-controlled input**: Talk to the assistant using natural speech.
- 🧠 **Query enhancement & planning**: Converts vague user input into executable programming plans.
- ⚙️ **Tool execution & file creation**: Writes real code and executes it on your machine.
- 📁 **Dynamic file generation**: Supports `create_file(...)` from within the plan.
- 🧾 **Execution summary**: Assistant describes what it has done, step-by-step.
- 🔊 **Text-to-Speech Output**: Speaks results back to you using OpenAI TTS.
- 💾 **MongoDB-based checkpointing**: Resumable LangGraph state flow.
---

## 🧠 Tech Stack
| Layer | Tools |
|-------|-------|
| LLM | `OpenAI GPT-4o`, `LangChain`, `LangGraph` |
| Voice I/O | `speech_recognition`, `openai.audio.speech`, `LocalAudioPlayer` |
| Workflow Graph | `LangGraph` |
| File Execution | `subprocess`, `run_command` tool |
| State Checkpointing | `MongoDB`, `MongoDBSaver` |
| Infra | `Docker`, `docker-compose` |

---

## 🗂️ Project Structure
```
├── main.py # Entry point: handles voice input/output and LangGraph execution
├── graph_windows.py # Defines LangGraph nodes, state machine, tools (For Windows)
├── graph.py # Defines LangGraph nodes, state machine, tools (For Mac)
├── docker-compose.yml # Spins up MongoDB locally for checkpointing
├── ai_solution/ # Directory for generated Python files
├── .env # OpenAI API Key and other secrets
```
---

## 🚀 Getting Started

1. Clone the repo
```
git clone https://github.com/ManeswarSahu11/cursorTalks
cd cursorTalks
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Add .env file
```
OPENAI_API_KEY=your_openai_api_key
```
4. Start MongoDB (via Docker)
```
docker-compose up -d
```
5. Run the assistant
```
python app/main.py
```
---
## 🗣️ How It Works
🎙️ You speak a programming task: “Create a Python script to reverse a string”
🧠 The assistant:
- Enhances the query
- Plans the steps
- Creates the file
- Executes it

🔊 The result is spoken back to you: “File created and executed successfully. Output: reversed string”

- Flow: enhance_query → create_plan → execute_step → tools → summary

🧪 Example Prompts
- “Make a Python file that sorts a list of numbers.”
- “Create a program that calculates factorial using recursion.”

---
📌 TODOs / Improvements
- Add confirmation before executing dangerous commands
- Add support for multiple languages (Python, C++, etc.)
- Build a web-based version using microphone API
- Logging interface for executed plans and tool calls

### 👨‍💻 Author: Maneswar Sahu
📄 MIT License – Feel free to use, modify and distribute!
