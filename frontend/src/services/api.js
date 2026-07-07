const API_URL = "http://localhost:8000/chat";

export const sendMessageToGateway = async (chatHistory) => {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // We pass the entire stateless array here, just like you designed!
      body: JSON.stringify({ messages: chatHistory }), 
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
};