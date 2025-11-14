# AI-Powered-Customer-Support-Chatbot-
Developed an NLP-driven chatbot capable of answering queries, improving engagement, and enhancing response efficiency.
# AI Customer Support Chatbot

A simple AI-powered customer support chatbot that gives automated responses and includes a Smart Query Queue & Retry System to handle API rate-limit problems.

# Features

- Gives AI-based answers to user queries

- Detects API rate-limit (429) errors

- Adds user queries to a queue

- Retries automatically after a delay

- Simple HTML chat interface

- Node.js backend handling all API calls

- Technologies Used

HTML, CSS, JavaScript

Node.js, Express

AI API (OpenAI/Gemini or any model)

# How It Works

User sends a message

Backend calls the AI API

If rate-limit occurs → message goes into queue

System retries after a few seconds

User receives the response automatically
