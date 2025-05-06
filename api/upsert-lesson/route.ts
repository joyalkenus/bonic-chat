import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from 'openai';
import { v4 as uuidv4 } from 'uuid';
import { NextResponse } from 'next/server';

// Initialize OpenAI client (for embeddings)
const openaiClient = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Initialize Pinecone client
const pc = new Pinecone({ 
  apiKey: process.env.PINECONE_API_KEY! 
});

// Access index with explicit host
const INDEX_NAME = process.env.PINECONE_INDEX_NAME!;
const INDEX_HOST = process.env.PINECONE_INDEX_HOST!;

if (!INDEX_NAME || !INDEX_HOST) {
  throw new Error('PINECONE_INDEX_NAME or PINECONE_INDEX_HOST environment variable is not set.');
}

const index = pc.index(INDEX_NAME, INDEX_HOST);
const pineconeNamespace = index.namespace('ns1'); // Default namespace

/**
 * Helper function to get embeddings from OpenAI
 * @param text Text to generate embedding for
 * @returns Array of embedding values
 */
async function getEmbedding(text: string): Promise<number[]> {
  // Basic cleaning: remove excessive newlines/whitespace
  const cleanedText = text.replace(/\s\s+/g, ' ').trim();
  if (!cleanedText) {
    throw new Error("Cannot generate embedding for empty text.");
  }
  try {
    const response = await openaiClient.embeddings.create({
      model: 'text-embedding-ada-002',
      input: cleanedText,
    });
    return response.data[0].embedding;
  } catch (error) {
    console.error("Error getting embedding:", error);
    throw new Error(`Failed to get embedding: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Interface for a single lesson upsert request
 */
interface LessonUpsertRequest {
  id?: string | number;    // Optional ID (will be generated if not provided)
  content: string;         // Required lesson content
  metadata?: Record<string, any>; // Optional additional metadata
}

/**
 * POST handler for upserting a single lesson
 */
export async function POST(req: Request) {
  try {
    // Parse request body
    const lessonData = await req.json() as LessonUpsertRequest;

    // Validate request contains required content field
    if (!lessonData.content || typeof lessonData.content !== 'string') {
      return NextResponse.json({ 
        error: 'Invalid request: content field is required and must be a string.' 
      }, { status: 400 });
    }

    // Generate embedding for lesson content
    const embedding = await getEmbedding(lessonData.content);
    
    // Use provided ID or generate a new one
    const lessonId = lessonData.id || uuidv4();
    
    // Convert any numeric IDs to strings for Pinecone compatibility
    const stringId = typeof lessonId === 'number' ? lessonId.toString() : lessonId;

    // Create vector with metadata
    const vectorToUpsert = {
      id: stringId,
      values: embedding,
      metadata: {
        content: lessonData.content, // Store original content in metadata
        ...(lessonData.metadata || {}), // Merge optional metadata
        lastUpdated: new Date().toISOString(),
      },
    };

    console.log(`Upserting lesson with ID: ${stringId}`);
    
    // Upsert the vector into Pinecone
    await pineconeNamespace.upsert([vectorToUpsert]);

    console.log(`Successfully upserted lesson with ID: ${stringId}`);

    return NextResponse.json({ 
      message: `Successfully upserted lesson with ID: ${stringId}`,
      id: stringId
    }, { status: 200 });

  } catch (error) {
    console.error('Error in upsert API:', error);
    return NextResponse.json(
      { error: `There was an error processing your request: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    );
  }
}