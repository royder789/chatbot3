import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import SQLChatMessageHistory

from sqlalchemy import create_engine

load_dotenv()

app = Flask(__name__, static_folder="static")

db_uri = f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
engine = create_engine(db_uri)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. You remember past conversations."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

def get_session_history(session_id):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=engine,
        table_name="message_store"
    )

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id")

    if not user_input or not session_id:
        return jsonify({"error": "Message and Session ID are required"}), 400

    try:
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return jsonify({"response": response.content})

    except Exception as e:
        print("ERROR:", e)  
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)