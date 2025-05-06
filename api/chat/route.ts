import { Pinecone } from '@pinecone-database/pinecone';
import { ChatOpenAI } from '@langchain/openai';
import { PineconeStore } from '@langchain/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { NextResponse } from 'next/server';
import { NextRequest } from 'next/server';
import { 
  createOpenAIToolsAgent,
  AgentExecutor
} from 'langchain/agents';
import { createRetrieverTool } from 'langchain/tools/retriever';
import { 
  ChatPromptTemplate, 
  MessagesPlaceholder, 
  HumanMessagePromptTemplate, 
  SystemMessagePromptTemplate 
} from '@langchain/core/prompts';
import { HumanMessage, AIMessage } from '@langchain/core/messages';

/**
 * Interface representing a chat session with an agent executor and chat history
 */
interface ChatSession {
  agentExecutor: AgentExecutor;
  chatHistory: (HumanMessage | AIMessage)[];
}

/**
 * Interface for the retriever tool with filter capabilities
 */
interface RetrieverToolWithFilter {
  name: string;
  description: string;
  retriever: {
    filter?: any;
    [key: string]: any;
  };
  [key: string]: any;
}

// Initialize Pinecone client
const pc = new Pinecone({ 
  apiKey: process.env.PINECONE_API_KEY || ''
});

// Index setup with environment variables
const INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'fusion360-lessons';
const INDEX_HOST = process.env.PINECONE_INDEX_HOST;

// Ensure the index host is provided
if (!INDEX_HOST) {
  throw new Error('PINECONE_INDEX_HOST environment variable is not set.');
}

// Connect to Pinecone index
const index = pc.index(INDEX_NAME, INDEX_HOST);

// In-memory conversation store for persistent chat sessions
const conversationHistories = new Map<string, ChatSession>();

/**
 * POST handler for chat functionality with Pinecone vector store integration
 * Supports filtering by lesson IDs to retrieve targeted content
 */
export async function POST(req: NextRequest) {
  try {
    // Extract request parameters with defaults
    const { messages, userId = 'default-user', lessonIds = [] } = await req.json();
    
    // Validate input messages
    if (!messages || !messages.length) {
      return NextResponse.json({ error: 'No messages found' }, { status: 400 });
    }
    
    // Extract the latest message content
    const latestMessage = messages[messages.length - 1];
    if (!latestMessage || !latestMessage.content) {
      return NextResponse.json({ error: 'Invalid message format' }, { status: 400 });
    }
    
    const userMessage = latestMessage.content;
    console.log(`Processing message for user ${userId}: "${userMessage}"`);
    
    // Retrieve existing user session or create a new one
    let userSession = conversationHistories.get(userId);
    
    // Create metadata filter if lesson IDs are provided
    // This filter is used to restrict search to specific documents
    const metadataFilter = lessonIds.length > 0 
      ? { id: { $in: lessonIds } }
      : undefined;
    
    console.log("Using metadata filter:", metadataFilter);
    
    // Initialize a new agent if this is a new session
    if (!userSession) {
      userSession = await createNewUserSession(metadataFilter);
      conversationHistories.set(userId, userSession);
    } else {
      // Update filters for existing session
      updateExistingSessionFilters(userSession, metadataFilter);
    }
    
    // Execute the agent with the user message and existing chat history
    const result = await userSession.agentExecutor.invoke({
      input: userMessage,
      chat_history: userSession.chatHistory
    });
    
    // Extract the response text from the result
    const responseText = result.output || "I couldn't generate a response. Please try again.";
    
    // Update the chat history with this interaction
    userSession.chatHistory.push(new HumanMessage(userMessage));
    userSession.chatHistory.push(new AIMessage(responseText));
    
    // Limit history length to prevent token overflow (keep last 10 messages)
    if (userSession.chatHistory.length > 10) {
      userSession.chatHistory = userSession.chatHistory.slice(-10);
    }
    
    // Return the response with debug information
    return NextResponse.json({
      response: responseText,
      debug: {
        usedRetrieval: result.intermediateSteps?.length > 0,
        appliedFilter: metadataFilter
      }
    });
    
  } catch (error) {
    // Error handling with proper error message extraction
    console.error('Error in chat API:', error);
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    
    return NextResponse.json(
      { error: `There was an error processing your request: ${errorMessage}` },
      { status: 500 }
    );
  }
}

/**
 * Creates a new user session with an agent executor and empty chat history
 * 
 * @param metadataFilter - Optional filter to restrict document search
 * @returns A new ChatSession with configured agent executor
 */
async function createNewUserSession(metadataFilter?: any): Promise<ChatSession> {
  // Initialize OpenAI embeddings for vector search
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });
  
  // Initialize the vector store with Pinecone and apply filters if provided
  const vectorStore = await PineconeStore.fromExistingIndex(
    embeddings,
    {
      pineconeIndex: index,
      namespace: 'ns1',
      textKey: 'content',
      filter: metadataFilter // Apply filter to vector store initialization
    }
  );
  
  // Create a retriever with the same filter
  const retriever = vectorStore.asRetriever({
    k: 3, // Number of documents to retrieve
    filter: metadataFilter
  });
  
  // Create the retriever tool for the agent to use
  const retrieverTool = createRetrieverTool(
    retriever,
    {
      name: "fusion360_search",
      description: "Search for information about Fusion 360. This tool should be used specifically for queries related to Fusion 360 CAD software - its features, workflows, or technical details. Only use this tool if the query is specifically about Fusion 360 or 3D modeling."
    }
  );
  
  // Initialize the LLM with appropriate temperature setting
  const llm = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
    temperature: 0, // Set to 0 for more deterministic responses
  });
  
  // Create the agent prompt template with system instructions and placeholders
  const prompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
      `You are an AI Teacher who is helping a student learn about various topics in robotics, here we are using Fusion 360 CAD software.

Your responses should be:
1. Always use the retrieved context to answer the question.
2. Include clear step-by-step instructions when applicable from the retrieved context.
3. Properly formatted in Markdown, preserving any image links from the context
4. Always include the image url in the response in mdx format.

IMPORTANT ABOUT IMAGES:
- Always preserve image URLs in your responses using proper Markdown image syntax: ![alt text](image_url)
- Do NOT modify image URLs - use them exactly as they appear in the context

Use your judgment about when to use the fusion360_search tool:
- Use the tool for any questions about Fusion 360 features, workflows, or technical details since we are following the curriculum.
- If you're unsure whether information exists in the knowledge base, it's better to try using the tool than to not use it.

Keep your responses natural and helpful. If you use the search tool, incorporate the information from the retrieved documents seamlessly into your answer.`
    ),
    // Add placeholder for chat history
    new MessagesPlaceholder({
      variableName: "chat_history",
    }),
    // Add human message template
    HumanMessagePromptTemplate.fromTemplate("{input}"),
    // Add agent scratchpad for tool usage
    new MessagesPlaceholder({
      variableName: "agent_scratchpad"
    })
  ]);
  
  // Create the OpenAI tools agent
  const agent = await createOpenAIToolsAgent({
    llm,
    tools: [retrieverTool],
    prompt
  });
  
  // Create the agent executor with the tool
  const agentExecutor = new AgentExecutor({
    agent,
    tools: [retrieverTool],
    verbose: true // Enable verbose mode for debugging
  });
  
  // Return new session with empty chat history
  return {
    agentExecutor,
    chatHistory: []
  };
}

/**
 * Updates the filter settings for an existing user session
 * 
 * @param userSession - The user's current chat session
 * @param metadataFilter - The filter to apply to document searches
 */
function updateExistingSessionFilters(userSession: ChatSession, metadataFilter?: any): void {
  // Skip if no filter is provided
  if (!metadataFilter) return;
  
  const tools = userSession.agentExecutor.tools;
  
  // Find the retriever tool by name
  const retrieverToolIndex = tools.findIndex(tool => tool.name === "fusion360_search");
  
  if (retrieverToolIndex !== -1) {
    // Get the current retriever tool and cast to our interface
    const currentRetrieverTool = tools[retrieverToolIndex] as unknown as RetrieverToolWithFilter;
    
    // Update the underlying retriever with the new filter if it exists
    if (currentRetrieverTool.retriever) {
      currentRetrieverTool.retriever.filter = metadataFilter;
    }
  }
}

/**
 * Example usage with specific lesson IDs:
 * 
 * POST request body:
 * {
 *   "messages": [{"role": "user", "content": "Tell me about Fusion 360 basics"}],
 *   "userId": "user123",
 *   "lessonIds": ["1001", "1002", "1003"]
 * }
 */