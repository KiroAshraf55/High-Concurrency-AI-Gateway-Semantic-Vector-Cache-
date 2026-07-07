import { useState } from 'react';
import { sendMessageToGateway } from '../services/api';

export const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (content) => {
    if (!content.trim()) return;

    const userMessage = { role: "user", content };
    // Create the new history array including the user's new message
    const updatedHistory = [...messages, userMessage];
    
    // Instantly show the user's message in the UI
    setMessages(updatedHistory);
    setIsLoading(true);

    try {
      // Send the full array to your Python Gateway
      const data = await sendMessageToGateway(updatedHistory);
      
      const aiMessage = { role: "assistant", content: data.ai_response };
      // Add the AI's response to the history
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = { role: "assistant", content: "Error: Could not connect to AI Gateway." };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return { messages, isLoading, sendMessage };
};