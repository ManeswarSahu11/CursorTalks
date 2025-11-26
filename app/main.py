import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from graph_windows import create_chat_graph
import asyncio
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

MONGODB_URI="mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "12"}} 

openai = AsyncOpenAI()

async def speak(text: str):
    """Convert text to speech and play it"""
    try:
        async with openai.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Speak in a cheerful and positive tone, like a helpful coding assistant.",
            response_format="pcm",
        ) as response:
            await LocalAudioPlayer().play(response)
    except Exception as e:
        print(f"Speech synthesis error: {e}")
        print(f"Text that would have been spoken: {text}")

def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)
        
        r = sr.Recognizer()

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            r.pause_threshold = 3  # Waits for 3 second pause

            print("🤖 AI Coding Assistant Ready!")
            print("Say something programming-related, or say 'exit' to quit.")
            
            while True:
                try:
                    print("\n🎤 Listening... (Say something!)")
                    audio = r.listen(source, timeout=10, phrase_time_limit=10)

                    print("🔄 Processing audio...")
                    speech_to_text = r.recognize_google(audio)
                    
                    print(f"👤 You Said: {speech_to_text}")
                    
                    # Check for exit command
                    if speech_to_text.lower().strip() in ['exit', 'quit', 'stop', 'goodbye']:
                        farewell_text = "Goodbye! Happy coding!"
                        print(f"🤖 {farewell_text}")
                        asyncio.run(speak(farewell_text))
                        break
                    
                    # Initialize state properly
                    initial_state = {
                        "messages": [{"role": "user", "content": speech_to_text}],
                        "enhanced_query": "",
                        "plan": [],
                        "current_step": 0,
                        "execution_summary": "",
                        "awaiting_confirmation": False,
                        "dangerous_command": ""
                    }
                    
                    print("🧠 Processing your request...")
                    
                    # Track execution phases
                    phases = ["🔍 Enhancing query", "📋 Creating plan", "⚡ Executing steps", "📝 Generating summary"]
                    phase_index = 0
                    
                    # Stream through the graph execution but only show final results
                    all_events = []
                    for event in graph.stream(initial_state, config, stream_mode="values"):
                        all_events.append(event)
                        
                        # Show progress without duplicates
                        if phase_index < len(phases):
                            print(f"   {phases[phase_index]}")
                            phase_index += 1
                    
                    # Only process the final event to avoid duplicates
                    if all_events:
                        final_event = all_events[-1]
                        
                        # Show plan if available
                        if "plan" in final_event and final_event["plan"] and final_event["plan"][0] != "REJECT_NON_PROGRAMMING":
                            print(f"📋 Execution Plan: {' → '.join(final_event['plan'])}")
                        
                        if "messages" in final_event and final_event["messages"]:
                            last_message = final_event["messages"][-1]
                            
                            # Print the message if it's from the AI
                            if hasattr(last_message, 'content') and last_message.content:
                                print(f"🤖 Result: {last_message.content}")
                                final_response = last_message.content
                            elif hasattr(last_message, 'pretty_print'):
                                last_message.pretty_print()
                                if hasattr(last_message, 'content'):
                                    final_response = last_message.content
                    
                    # Speak the final response
                    if final_response:
                        print("🔊 Speaking response...")
                        asyncio.run(speak(final_response))
                    
                except sr.WaitTimeoutError:
                    print("⏰ No speech detected, continuing to listen...")
                    continue
                except sr.UnknownValueError:
                    error_msg = "Sorry, I couldn't understand what you said. Could you please repeat?"
                    print(f"🤖 {error_msg}")
                    asyncio.run(speak(error_msg))
                    continue
                except sr.RequestError as e:
                    error_msg = f"Could not request results from speech recognition service; {e}"
                    print(f"❌ Error: {error_msg}")
                    continue
                except KeyboardInterrupt:
                    farewell_text = "Session ended by user. Goodbye!"
                    print(f"\n🤖 {farewell_text}")
                    asyncio.run(speak(farewell_text))
                    break
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {e}"
                    print(f"❌ Error: {error_msg}")
                    asyncio.run(speak("Sorry, I encountered an error. Please try again."))
                    continue

if __name__ == "__main__":
    main()