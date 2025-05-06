# Learning Platform API Documentation

This API provides two main endpoints for managing and interacting with educational content:

1. **Lesson Upsert API** - For adding or updating lessons in the vector database
2. **Chat API** - For retrieving and interacting with lesson content using a conversational interface

## Prerequisites

Before using these APIs, you need to set up the following environment variables:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_INDEX_HOST=your_pinecone_host.pinecone.io
```

## 1. Lesson Upsert API

This endpoint allows you to add or update individual lessons in the Pinecone vector database.

### Endpoint

```
POST /api/upsert-lesson
```

### Request Format

```json
{
  "id": "1011",                 // Optional: Will be auto-generated if not provided
  "content": "## Lesson content with Markdown formatting",
  "metadata": {                 // Optional: Additional metadata for filtering
    "topic": "Example Topic",
    "difficulty": "Beginner",
    "lesson_type": "Tutorial"
  }
}
```

### Response Format

```json
{
  "message": "Successfully upserted lesson with ID: 1011",
  "id": "1011"
}
```

### Example Usage

```bash
curl -X POST http://localhost:3000/api/upsert-lesson \
  -H "Content-Type: application/json" \
  -d '{
    "id": "1011",
    "content": "## Objective\nLearn about an important topic.\n\n## Overview\nThis lesson covers key concepts and practical applications.\n\n![Reference Image](https://example.com/image.jpg)",
    "metadata": {
      "topic": "Main Topic",
      "difficulty": "Intermediate",
      "lesson_type": "Tutorial"
    }
  }'
```

### Notes

- The API automatically generates embeddings using OpenAI's text-embedding-ada-002 model
- Numeric IDs are converted to strings for Pinecone compatibility
- The content is stored in its original form in the metadata for retrieval
- A timestamp is automatically added to the metadata

## 2. Chat API with Filtered Retrieval

This endpoint provides a conversational interface to interact with the lessons stored in the database, with the ability to filter by specific lesson IDs.

### Endpoint

```
POST /api/chat
```

### Request Format

```json
{
  "messages": [
    {"role": "user", "content": "Tell me about the topic"}
  ],
  "userId": "user123",         // Optional: For maintaining conversation history
  "lessonIds": ["1001", "1002", "1003"]  // Optional: For filtering specific lessons
}
```

### Response Format

```json
{
  "response": "Here's information about the topic you asked about...",
  "debug": {
    "usedRetrieval": true,
    "appliedFilter": {"id": {"$in": ["1001", "1002", "1003"]}}
  }
}
```

### Example Usage

```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me about the key concepts"}],
    "userId": "user123",
    "lessonIds": ["1011"]
  }'
```

### Features

- **Conversation History**: Maintains chat history for each user ID
- **Filtered Retrieval**: Can restrict searches to specific lesson IDs
- **Contextual Responses**: Uses LangChain agents to provide relevant responses
- **Image Handling**: Preserves image URLs from the lessons in responses
- **Dynamic Tool Usage**: Uses retrieval tools based on the query context

## Implementation Details

### Upsert Process

1. The lesson content is cleaned and converted to an embedding vector
2. The vector and metadata are stored in the Pinecone database
3. The original content is preserved in the metadata for retrieval

### Chat Process

1. User messages are processed through an agent with retrieval capabilities
2. If lesson IDs are provided, searches are filtered to only include those lessons
3. The agent uses OpenAI to generate responses based on retrieved content
4. Chat history is maintained for each user within a server-side map
5. Responses include original formatting and images from the lessons

## Best Practices

1. **Lesson Structure**: Format lesson content with Markdown for best results
2. **Consistent IDs**: Use a consistent ID scheme for easier filtering
3. **Detailed Metadata**: Include metadata for more precise filtering options
4. **Clear Questions**: For best retrieval, ask specific questions
5. **User IDs**: Use consistent user IDs to maintain conversation context

## Error Handling

Both APIs include detailed error handling with appropriate status codes:

- 400: Invalid request format or missing required fields
- 500: Server errors during processing (embedding generation, database operations)

## Deployment

Deploy these endpoints in your Next.js application by placing them in the appropriate route files:

- `/app/api/upsert-lesson/route.ts` - For the lesson upsert API
- `/app/api/chat/route.ts` - For the chat API

## Customizing the System Prompt

You can customize the chat API's behavior by modifying the system prompt in the `createNewUserSession` function. The prompt determines how the AI responds to questions and what context it has about the content.
