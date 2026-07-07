import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import '../styles/ChatInterface.css';
import ReactMarkdown from 'react-markdown';

const ChatInterface = () => {
  const { messages, isLoading, sendMessage } = useChat();
  const [inputValue, setInputValue] = useState("");
  const messagesEndRef = useRef(null);

  // Auto-scroll to the newest message
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = (e) => {
    e.preventDefault();
    sendMessage(inputValue);
    setInputValue("");
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h2>AI Gateway Cache v3</h2>
        <span className="status">Status: 🟢 Online</span>
      </header>

      <div className="chat-window">
        {messages.length === 0 && (
          <div className="empty-state">Send a message to start caching!</div>
        )}
        
        {messages.map((msg, index) => (
          <div key={index} className={`message-wrapper ${msg.role}`}>
            <div className={`message-bubble ${msg.role}`}>
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message-wrapper assistant">
            <div className="message-bubble typing">AI is thinking...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-area" onSubmit={handleSend}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a question..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !inputValue.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;