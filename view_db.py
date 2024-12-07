import sqlite3

def view_chat_history():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT * FROM chat_history ORDER BY timestamp DESC")
    history = c.fetchall()
    conn.close()
    
    if not history:
        print("No chat history found.")
    else:
        for entry in history:
            print(f"Timestamp: {entry[1]}")
            print(f"User: {entry[2]}")
            print(f"AI: {entry[3]}")
            print("-" * 50)

if __name__ == "__main__":
    view_chat_history()
