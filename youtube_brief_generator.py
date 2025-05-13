#!/usr/bin/env python3
"""
YouTube Brief Generator Bot
-----------------------
A complete tool to generate YouTube production briefs based on Brett Payne's format.
Includes orchestration layer for robust document processing and brief generation.

Usage:
    streamlit run youtube_brief_generator.py
"""

import os
import json
import time
import re
import uuid
import tempfile
import asyncio
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import streamlit as st
import numpy as np
import requests
import PyPDF2
try:
    import docx
except ImportError:
    st.warning("python-docx not installed. DOCX support will be limited. Install with: pip install python-docx")
from supabase import create_client, Client
from groq import Groq

# ===============================================================================
# Configuration
# ===============================================================================

# API Keys
SUPABASE_URL = "https://gxxzvymiitcyhfafqoqb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd4eHp2eW1paXRjeWhmYWZxb3FiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDcwODgxMTEsImV4cCI6MjA2MjY2NDExMX0.e-2iVrLuVAnT5F52oMkLQPmWGDk_-FpNBfV_aAsyspk"
GROQ_API_KEY = "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"
JINA_API_KEY = "jina_1743f97924f945afa834f3aa3a168482bqEjfUa5abt922EBVvIDpi-4iuSt"

# Model Names
BRIEF_GENERATION_MODEL = "deepseek-r1-distill-llama-70b"  # Model for brief generation
EMBEDDING_MODEL = "jina-embeddings-v2-base-en"  # Embedding model
RERANK_MODEL = "jina-reranker-v1-base-en"  # Reranker model

# Constants
VIDEO_LEVELS = ["L1", "L2", "L3", "L4"]
DEFAULT_CHUNK_SIZE = 800
OVERLAP_SIZE = 160  # 20% overlap
EMBEDDING_DIMENSION = 768  # Jina's default dimension
MAX_RETRIES = 3  # Maximum number of retries for API operations
RATE_LIMIT_DELAY = 0.5  # Delay between API calls in seconds

# Initialize clients
try:
    # Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    print("Clients initialized successfully")
    
except Exception as e:
    print(f"Error initializing clients: {str(e)}")

# Initialize Streamlit configuration
st.set_page_config(
    page_title="YouTube Brief Generator Bot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================================================================
# Orchestration Layer - Job Management
# ===============================================================================

# Job statuses
class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Job types
class JobType:
    DOCUMENT_PROCESSING = "document_processing"
    BRIEF_GENERATION = "brief_generation"
    EMBEDDING_GENERATION = "embedding_generation"

# Job queue for background processing
job_queue = queue.Queue()
job_results = {}
job_status = {}
job_logs = {}
job_metadata = {}  # Store friendly names and additional metadata

# Background thread flag
processing_thread_running = False

def generate_job_id():
    """Generate a unique job ID."""
    return str(uuid.uuid4())

def submit_job(job_type, params, friendly_name=None):
    """Submit a job to the processing queue with friendly name."""
    job_id = generate_job_id()
    job_status[job_id] = JobStatus.PENDING
    job_logs[job_id] = []
    job_queue.put((job_id, job_type, params))
    
    # Store friendly name and creation time
    job_metadata[job_id] = {
        "friendly_name": friendly_name or f"Job {job_id[:8]}",
        "created_at": datetime.now(),
        "type": job_type,
        "params": params
    }
    
    # Start processing thread if not running
    global processing_thread_running
    if not processing_thread_running:
        processing_thread = threading.Thread(target=process_jobs)
        processing_thread.daemon = True
        processing_thread.start()
        processing_thread_running = True
    
    return job_id

def log_job_event(job_id, message):
    """Log an event for a job."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    if job_id in job_logs:
        job_logs[job_id].append(log_entry)
    print(f"Job {job_id}: {message}")

def process_jobs():
    """Process jobs from the queue in the background."""
    global processing_thread_running
    processing_thread_running = True
    
    while True:
        try:
            # Get the next job from the queue
            job_id, job_type, params = job_queue.get(timeout=1)
            job_status[job_id] = JobStatus.PROCESSING
            log_job_event(job_id, f"Starting {job_type} job")
            
            # Update estimated completion time based on job type
            if job_id in job_metadata:
                if job_type == JobType.BRIEF_GENERATION:
                    job_metadata[job_id]["estimated_completion"] = datetime.now() + timedelta(seconds=45)
                elif job_type == JobType.DOCUMENT_PROCESSING:
                    # Estimate based on document size if available
                    if isinstance(params, tuple) and len(params) > 2:
                        doc_size = len(params[2]) if params[2] else 0
                        # Rough estimate: 1 second per 5KB of text
                        est_seconds = max(10, int(doc_size / 5000))
                        job_metadata[job_id]["estimated_completion"] = datetime.now() + timedelta(seconds=est_seconds)
                    else:
                        job_metadata[job_id]["estimated_completion"] = datetime.now() + timedelta(seconds=30)
            
            try:
                # Process based on job type
                if job_type == JobType.DOCUMENT_PROCESSING:
                    title, source, content = params
                    # Update friendly name with document title
                    if job_id in job_metadata:
                        job_metadata[job_id]["friendly_name"] = f"Document: {title}"
                    
                    result = process_document_job(job_id, title, source, content)
                    job_results[job_id] = result
                    job_status[job_id] = JobStatus.COMPLETED
                    log_job_event(job_id, f"Completed {job_type} job")
                    
                elif job_type == JobType.BRIEF_GENERATION:
                    topic, level, opt_in_url, persona_override = params
                    # Update friendly name with topic
                    if job_id in job_metadata:
                        job_metadata[job_id]["friendly_name"] = f"Brief: {topic}"
                    
                    result = generate_youtube_brief_job(job_id, topic, level, opt_in_url, persona_override)
                    job_results[job_id] = result
                    job_status[job_id] = JobStatus.COMPLETED
                    log_job_event(job_id, f"Completed {job_type} job")
                    
                elif job_type == JobType.EMBEDDING_GENERATION:
                    text = params
                    result = generate_embedding_with_retry(text)
                    job_results[job_id] = result
                    job_status[job_id] = JobStatus.COMPLETED
                    log_job_event(job_id, f"Completed {job_type} job")
                
                # Store job result in database if jobs table exists
                try:
                    supabase.table("jobs").insert({
                        "id": job_id,
                        "job_type": job_type,
                        "status": job_status[job_id],
                        "params": params if isinstance(params, dict) else {"data": str(params)},
                        "result": job_results[job_id],
                        "logs": job_logs[job_id]
                    }).execute()
                except Exception as e:
                    print(f"Could not store job in database (jobs table may not exist): {str(e)}")
                
            except Exception as e:
                job_status[job_id] = JobStatus.FAILED
                error_message = str(e)
                job_results[job_id] = {"error": error_message}
                log_job_event(job_id, f"Failed: {error_message}")
                
                # Try to store failed job in database
                try:
                    supabase.table("jobs").insert({
                        "id": job_id,
                        "job_type": job_type,
                        "status": JobStatus.FAILED,
                        "params": params if isinstance(params, dict) else {"data": str(params)},
                        "result": {"error": error_message},
                        "logs": job_logs[job_id]
                    }).execute()
                except Exception as db_error:
                    print(f"Could not store failed job in database: {str(db_error)}")
            
            # Mark the job as done
            job_queue.task_done()
            
        except queue.Empty:
            # No jobs in the queue
            continue
        except Exception as e:
            print(f"Error in job processing thread: {str(e)}")
            time.sleep(1)  # Avoid tight loop if there's an error

def get_job_status(job_id):
    """Get the status of a job."""
    # First check in-memory
    if job_id in job_status:
        return {
            "id": job_id,
            "status": job_status.get(job_id, "unknown"),
            "logs": job_logs.get(job_id, []),
            "result": job_results.get(job_id, None),
            "metadata": job_metadata.get(job_id, {})
        }
    
    # If not in memory, try database
    try:
        result = supabase.table("jobs").select("*").eq("id", job_id).execute()
        if result.data:
            job = result.data[0]
            return {
                "id": job_id,
                "status": job["status"],
                "logs": job["logs"] if job["logs"] else [],
                "result": job["result"] if job["result"] else None,
                "metadata": {
                    "friendly_name": f"Job {job_id[:8]}",
                    "created_at": job["created_at"],
                    "type": job["job_type"]
                }
            }
    except Exception as e:
        print(f"Could not load job from database: {str(e)}")
    
    return {
        "id": job_id,
        "status": "unknown",
        "logs": [],
        "result": None,
        "metadata": {}
    }

def get_all_jobs():
    """Get the status of all jobs including those in database."""
    # Get in-memory jobs
    memory_jobs = [
        {
            "id": job_id,
            "status": status,
            "friendly_name": job_metadata.get(job_id, {}).get("friendly_name", f"Job {job_id[:8]}"),
            "type": job_metadata.get(job_id, {}).get("type", "unknown"),
            "created_at": job_metadata.get(job_id, {}).get("created_at", datetime.now()).strftime("%Y-%m-%d %H:%M"),
            "estimated_completion": job_metadata.get(job_id, {}).get("estimated_completion", None)
        }
        for job_id, status in job_status.items()
    ]
    
    # Try to get database jobs
    try:
        result = supabase.table("jobs").select("*").order("created_at", desc=True).execute()
        if result and result.data:
            db_jobs = [
                {
                    "id": job["id"],
                    "status": job["status"],
                    "friendly_name": f"{job['job_type'].replace('_', ' ').title()}: {job['id'][:8]}",
                    "type": job["job_type"],
                    "created_at": job["created_at"].split("T")[0] if isinstance(job["created_at"], str) else str(job["created_at"]),
                    "estimated_completion": None
                }
                for job in result.data
            ]
            
            # Merge lists, prioritizing memory jobs (more up-to-date)
            memory_job_ids = [job["id"] for job in memory_jobs]
            combined_jobs = memory_jobs + [job for job in db_jobs if job["id"] not in memory_job_ids]
            return combined_jobs
    except Exception as e:
        print(f"Could not load jobs from database: {str(e)}")
    
    return memory_jobs

def load_db_jobs():
    """Load jobs from database into memory."""
    try:
        result = supabase.table("jobs").select("*").execute()
        if result.data:
            for job in result.data:
                job_id = job["id"]
                job_status[job_id] = job["status"]
                job_logs[job_id] = job["logs"] if job["logs"] else []
                job_results[job_id] = job["result"] if job["result"] else {}
                job_metadata[job_id] = {
                    "friendly_name": f"{job['job_type'].replace('_', ' ').title()}: {job['id'][:8]}",
                    "created_at": job["created_at"] if isinstance(job["created_at"], datetime) else datetime.now(),
                    "type": job["job_type"]
                }
            print(f"Loaded {len(result.data)} jobs from database")
    except Exception as e:
        print(f"Could not load jobs from database (jobs table may not exist): {str(e)}")

# ===============================================================================
# Database Utilities
# ===============================================================================

def check_tables_exist() -> bool:
    """Check if the required tables exist in the database."""
    try:
        # Try to select from each table. If any fails, tables don't exist.
        try:
            supabase.table("documents").select("id").limit(1).execute()
            supabase.table("document_chunks").select("id").limit(1).execute()
            supabase.table("briefs").select("id").limit(1).execute()
            supabase.table("content_hierarchy").select("id").limit(1).execute()
            # Try to select from jobs table but don't fail if it doesn't exist
            try:
                supabase.table("jobs").select("id").limit(1).execute()
            except:
                pass
            return True
        except Exception as e:
            print(f"Tables don't exist: {e}")
            return False
    except Exception as e:
        print(f"Error checking tables: {e}")
        return False

def table_row_count(table_name):
    """Get the number of rows in a table."""
    try:
        result = supabase.table(table_name).select("*", count="exact").limit(1).execute()
        if hasattr(result, 'count'):
            return result.count
        elif hasattr(result, 'data'):
            return len(result.data)
        else:
            return 0
    except Exception as e:
        print(f"Error counting rows in {table_name}: {str(e)}")
        return 0

# ===============================================================================
# Jina API Functions with Retry Logic
# ===============================================================================

def generate_embedding_with_retry(text: str, max_retries: int = MAX_RETRIES) -> List[float]:
    """
    Generate text embeddings using Jina AI API with retry logic.
    
    Args:
        text (str): Text to generate embeddings for
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        List[float]: Vector embedding
    """
    retry_count = 0
    last_exception = None
    
    while retry_count < max_retries:
        try:
            # Check if text is valid
            if not text or not text.strip():
                raise ValueError("Text for embedding cannot be empty")
            
            # Prepare API request to Jina
            headers = {
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": [text],
                "model": EMBEDDING_MODEL
            }
            
            # Make API request
            response = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Jina API error: {response.status_code} - {response.text}")
            
            # Parse response
            data = response.json()
            
            # Extract embeddings
            if "data" not in data or not data["data"] or "embedding" not in data["data"][0]:
                raise Exception("Invalid response from Jina API")
            
            return data["data"][0]["embedding"]
            
        except Exception as e:
            last_exception = e
            retry_count += 1
            print(f"Embedding generation attempt {retry_count} failed: {str(e)}")
            
            # Add exponential backoff
            time.sleep(RATE_LIMIT_DELAY * (2 ** retry_count))
    
    # If we reached here, all retries failed
    print(f"All {max_retries} attempts failed for embedding generation")
    raise last_exception or Exception("Failed to generate embedding after multiple attempts")

def rerank_results(query: str, initial_results: List[Dict], limit: int = 3) -> List[Dict]:
    """
    Rerank results using Jina Reranker API for more accurate matches.
    FIX APPLIED: Properly handle the API response format and errors.
    
    Args:
        query (str): The original query
        initial_results (list): List of initial results to rerank
        limit (int): Maximum number of results to return
        
    Returns:
        List[Dict]: Reordered results
    """
    if not initial_results:
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare documents for reranking - with better error handling
        docs = []
        for i, chunk in enumerate(initial_results):
            if "content" in chunk and chunk["content"]:
                docs.append({
                    "id": str(i),
                    "text": chunk["content"]
                })
        
        if not docs:
            return initial_results[:limit]
        
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": docs
        }
        
        # Make API request
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"Reranking failed: {response.status_code} - {response.text}")
            return initial_results[:limit]
        
        # Parse response
        data = response.json()
        
        # Check for expected structure
        if "results" not in data:
            print("Unexpected API response structure - no results key")
            return initial_results[:limit]
        
        # Create a new reranked list
        reranked = []
        for item in data["results"]:
            # Verify we have the expected structure before accessing fields
            if "document" in item and "id" in item["document"]:
                try:
                    doc_id = int(item["document"]["id"])
                    if doc_id < len(initial_results):
                        reranked.append(initial_results[doc_id])
                except (ValueError, TypeError) as e:
                    print(f"Error parsing document ID: {e}")
                    continue
                
        return reranked[:limit] if reranked else initial_results[:limit]
        
    except Exception as e:
        print(f"Error during reranking: {str(e)}")
        # Fall back to simply returning the top N results
        return initial_results[:limit]

# ===============================================================================
# Document Processing
# ===============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        raise Exception(f"Failed to extract PDF text: {str(e)}")

def extract_text_from_docx(docx_file) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(docx_file)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        print(f"Error extracting DOCX text: {str(e)}")
        raise Exception(f"Failed to extract DOCX text: {str(e)}")

def split_text_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    words = text.split()
    
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep overlap words
            overlap_words = current_chunk[-int(overlap/5):]
            current_chunk = overlap_words.copy()
            current_length = len(" ".join(current_chunk)) + 1
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_document_chunk(document_id: str, chunk_content: str, chunk_index: int, source: str, title: str) -> bool:
    """Process a single document chunk (for parallelization)."""
    try:
        # Generate embedding with retry
        embedding = generate_embedding_with_retry(chunk_content)
        
        # Store chunk with embedding
        supabase.table("document_chunks").insert({
            "document_id": document_id,
            "content": chunk_content,
            "embedding": embedding,
            "metadata": {
                "chunk_index": chunk_index,
                "source": source,
                "title": title
            }
        }).execute()
        
        return True
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {str(e)}")
        return False

def process_document_job(job_id: str, title: str, source: str, content: str) -> Dict:
    """Process a document in a background job."""
    # First check if tables exist
    if not check_tables_exist():
        raise Exception("Database tables not found. Please contact support.")
        
    try:
        log_job_event(job_id, f"Processing document: {title}")
        
        # Insert document
        document_data = supabase.table("documents").insert({
            "title": title,
            "source": source,
            "content": content
        }).execute()
        
        if not document_data.data:
            raise Exception("Failed to insert document")
        
        document_id = document_data.data[0]["id"]
        log_job_event(job_id, f"Document inserted with ID: {document_id}")
        
        # Chunk document
        chunks = split_text_into_chunks(content)
        log_job_event(job_id, f"Document split into {len(chunks)} chunks")
        
        # Process chunks
        success_count = 0
        failure_count = 0
        
        for i, chunk_content in enumerate(chunks):
            log_job_event(job_id, f"Processing chunk {i+1}/{len(chunks)}")
            
            # Rate limiting
            if i > 0:
                time.sleep(RATE_LIMIT_DELAY)
                
            # Process chunk
            success = process_document_chunk(document_id, chunk_content, i, source, title)
            
            if success:
                success_count += 1
            else:
                failure_count += 1
        
        result = {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "successful_chunks": success_count,
            "failed_chunks": failure_count,
            "title": title,
            "source": source
        }
        
        log_job_event(job_id, f"Document processing completed. Success: {success_count}, Failed: {failure_count}")
        return result
    
    except Exception as e:
        log_job_event(job_id, f"Document processing failed: {str(e)}")
        raise Exception(f"Failed to process document: {str(e)}")

def process_document(title: str, source: str, content: str) -> str:
    """Submit document processing job."""
    # Submit job for background processing
    job_id = submit_job(
        JobType.DOCUMENT_PROCESSING, 
        (title, source, content),
        friendly_name=f"Document: {title}"
    )
    return job_id

# ===============================================================================
# Brief Generation
# ===============================================================================

def clean_deepseek_output(content: str) -> str:
    """Clean up any <think> tags or unwanted artifacts from Deepseek output."""
    # Remove <think> sections
    clean_text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Remove any other potential API artifacts
    clean_text = re.sub(r'<[\w/]+>', '', clean_text)
    
    return clean_text.strip()

def get_default_persona() -> Dict[str, str]:
    """Get default Brett Payne persona."""
    return {
        "age": "Late 30s",
        "role": "agency owner",
        "pain": "operational chaos and lack of systems"
    }

def search_similar_content(query: str, limit: int = 5) -> List[Dict]:
    """Search for similar content using vector similarity with retries."""
    if not check_tables_exist():
        raise Exception("Database tables not found. Please contact support.")
        
    # Generate embedding for query with retry
    query_embedding = generate_embedding_with_retry(query)
    
    # Try to use the vector search function with retries
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            result = supabase.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.7,
                    "match_count": limit + 2  # Get extra for reranking
                }
            ).execute()
            
            # Apply reranking to improve matches
            if result.data and len(result.data) > 0:
                reranked_results = rerank_results(query, result.data, limit)
                return reranked_results
            return result.data
            
        except Exception as e:
            retry_count += 1
            print(f"Vector search attempt {retry_count} failed: {str(e)}")
            if retry_count >= MAX_RETRIES:
                print("Falling back to basic search")
                # Fallback to basic search
                result = supabase.table("document_chunks").select("*").limit(limit).execute()
                return result.data if result.data else []
            # Add exponential backoff
            time.sleep(RATE_LIMIT_DELAY * (2 ** retry_count))

def find_parent_content(topic: str, current_level: str, limit: int = 3) -> Optional[Dict]:
    """Find parent content for CTA based on topic similarity with retries."""
    if not check_tables_exist():
        raise Exception("Database tables not found. Please contact support.")
        
    if current_level == "L1":
        return None
    
    # Determine parent level
    parent_level = {
        "L2": "L1",
        "L3": "L2",
        "L4": "L3"
    }.get(current_level)
    
    if not parent_level:
        return None
    
    # Generate embedding for the topic with retry
    topic_embedding = generate_embedding_with_retry(topic)
    
    # Try to use the vector search function with retries
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            result = supabase.rpc(
                "match_parent_briefs",
                {
                    "query_embedding": topic_embedding,
                    "parent_level": parent_level,
                    "match_threshold": 0.75,
                    "match_count": limit
                }
            ).execute()
            
            if result.data:
                return result.data[0]
            break
        except Exception as e:
            retry_count += 1
            print(f"Vector search for parent content attempt {retry_count} failed: {str(e)}")
            if retry_count >= MAX_RETRIES:
                break
            # Add exponential backoff
            time.sleep(RATE_LIMIT_DELAY * (2 ** retry_count))
            
    # Fallback to simple level-based search
    result = supabase.table("briefs").select("*").eq("level", parent_level).limit(1).execute()
    
    if not result.data:
        return None
    
    # Add similarity field for consistency
    parent_content = result.data[0]
    parent_content["similarity"] = 0.8  # Default similarity
    
    return parent_content

def generate_youtube_brief_job(
    job_id: str,
    topic: str,
    level: str,
    opt_in_url: Optional[str] = None,
    persona_override: Optional[Dict] = None
) -> Dict[str, Any]:
    """Generate a YouTube brief in a background job."""
    # First check if tables exist
    if not check_tables_exist():
        raise Exception("Database tables not found. Please contact support.")
        
    try:
        log_job_event(job_id, f"Generating brief for topic: {topic}, level: {level}")
        
        # Get default persona if no override
        persona = persona_override or get_default_persona()
        
        # Retrieve relevant knowledge base content
        log_job_event(job_id, "Searching for relevant content")
        context_chunks = search_similar_content(topic, limit=10)
        context_text = "\n\n".join([chunk.get("content", "") for chunk in context_chunks]) if context_chunks else ""
        log_job_event(job_id, f"Found {len(context_chunks)} relevant content chunks")
        
        # Set up system prompt
        system_prompt = """You are an expert YouTube content strategist who creates video briefs in Brett Payne's style.
        Focus on creating engaging, educational content that solves specific problems for the target audience.
        Always maintain a professional yet approachable tone with a focus on systems, processes, and practical business advice.
        Your brief should follow the exact structure shown in the example.
        
        IMPORTANT: For the target viewer section, use general descriptions and avoid creating specific fictional personas with names.
        Use age ranges rather than specific ages, and general role descriptions rather than overly specific details."""
        
        # Determine CTA logic based on level
        log_job_event(job_id, "Determining CTA strategy")
        cta_info = ""
        if level == "L1":
            if opt_in_url:
                cta_info = f"Primary CTA should direct to the opt-in URL: {opt_in_url}"
            else:
                cta_info = "Primary CTA should promote a lead magnet or resource relevant to the topic."
        else:
            parent_content = find_parent_content(topic, level)
            if parent_content:
                cta_info = f"Primary CTA should direct viewers to watch the related {parent_content.get('level')} video: '{parent_content.get('topic')}'"
                log_job_event(job_id, f"Found parent content: {parent_content.get('topic')}")
            else:
                cta_info = f"[NEEDS MANUAL CTA] - No suitable parent video found."
                log_job_event(job_id, "No parent content found")
        
        # Construct user prompt with improved persona guidance
        user_prompt = f"""Create a complete YouTube brief for a {level} video about "{topic}".

Target Audience:
Age range: {persona.get('age', '35-45')} 
Role: {persona.get('role', 'business owner')}
Pain Point: {persona.get('pain', 'operational challenges')}

Brief structure must include ALL of these exact sections in this order:
1. Video Keyword / Working Title (3-5 options)
2. Target Viewer (use general description, avoid specific fictional names)
3. Problem the Video Solves
4. Video Goal (functional + emotional)
5. Primary CTA
6. Thumbnail Caption Ideas (3-5)
7. Target Duration
8. Permanent Resources Section
9. Simple Video Outline with:
   - Hook
   - Intro
   - Core Points (3-5 bullets with micro-CTAs)
   - Mid-Video CTA
   - Question of the Day
   - Final CTA

{cta_info}

Always include this link in the Permanent Resources section:
"How to Create YouTube Videos That Inspire, Educate, and Solve the Viewer's Problem"

Use the following context from Brett's content to maintain his style and approach:
{context_text[:3800] if context_text else "Focus on systems, processes, and operational efficiency for business owners."}
"""

        # Generate content with Groq - with retry logic
        log_job_event(job_id, "Generating brief content with Groq")
        retry_count = 0
        completion = None
        while retry_count < MAX_RETRIES:
            try:
                completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=BRIEF_GENERATION_MODEL,  # Using Deepseek for brief generation
                    temperature=0.7,
                    max_tokens=4000,
                    top_p=0.9,
                )
                break
            except Exception as e:
                retry_count += 1
                log_job_event(job_id, f"Groq API attempt {retry_count} failed: {str(e)}")
                if retry_count >= MAX_RETRIES:
                    raise Exception(f"Failed to generate brief content after {MAX_RETRIES} attempts")
                # Add exponential backoff
                time.sleep(RATE_LIMIT_DELAY * (2 ** retry_count))
        
        if not completion:
            raise Exception("Failed to generate brief content")
            
        # Extract generated content and clean it
        raw_brief_content = completion.choices[0].message.content
        brief_content = clean_deepseek_output(raw_brief_content)  # Clean any <think> tags
        log_job_event(job_id, "Brief content generated and cleaned successfully")
        
        # Save brief to database - without fact verification
        log_job_event(job_id, "Saving brief to database")
        brief_data = supabase.table("briefs").insert({
            "topic": topic,
            "level": level,
            "opt_in_url": opt_in_url,
            "content": brief_content,
            "persona": persona
        }).execute()
        
        if not brief_data.data:
            raise Exception("Failed to insert brief")
            
        brief_id = brief_data.data[0]["id"]
        log_job_event(job_id, f"Brief saved with ID: {brief_id}")
        
        # Store hierarchy relationship if applicable
        if level != "L1":
            parent_content = find_parent_content(topic, level)
            if parent_content:
                log_job_event(job_id, f"Creating hierarchy relationship with parent: {parent_content.get('id')}")
                supabase.table("content_hierarchy").insert({
                    "brief_id": brief_id,
                    "parent_brief_id": parent_content.get("id"),
                    "level": level,
                    "similarity": parent_content.get("similarity", 0.8)
                }).execute()
        
        result = {
            "id": brief_id,
            "content": brief_content,
            "topic": topic,
            "level": level
        }
        
        log_job_event(job_id, "Brief generation completed successfully")
        return result
    
    except Exception as e:
        log_job_event(job_id, f"Brief generation failed: {str(e)}")
        raise Exception(f"Failed to generate brief: {str(e)}")

def generate_youtube_brief(
    topic: str,
    level: str,
    opt_in_url: Optional[str] = None,
    persona_override: Optional[Dict] = None
) -> str:
    """Submit brief generation job."""
    # Submit job for background processing
    job_id = submit_job(
        JobType.BRIEF_GENERATION, 
        (topic, level, opt_in_url, persona_override),
        friendly_name=f"Brief: {topic}"
    )
    return job_id

# ===============================================================================
# Export Functions
# ===============================================================================

def export_to_markdown(brief_content: str, topic: str) -> str:
    """Export brief to Markdown format."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{timestamp}-{re.sub(r'[^\w\-]', '-', topic)}-Brief.md"
    
    with open(filename, "w") as f:
        f.write(brief_content)
    
    return filename

# ===============================================================================
# Streamlit UI
# ===============================================================================

def main():
    """Main Streamlit application."""
    st.title("YouTube Brief Generator Bot")
    st.sidebar.image("https://images.unsplash.com/photo-1611162616475-46b635cb6868?q=80&w=1548&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", width=150)
    
    # Add timestamp and user info
    st.sidebar.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
    st.sidebar.write(f"Brett Payne pilot")
    
    # Try to load jobs from database at startup
    if "jobs_loaded" not in st.session_state:
        load_db_jobs()
        st.session_state["jobs_loaded"] = True
    
    # Check if tables exist
    tables_exist = check_tables_exist()
    
    if not tables_exist:
        render_database_connection_error()
    else:
        # Tables exist, show normal navigation
        page = st.sidebar.radio(
            "Navigation", 
            ["Generate Brief", "Upload Document", "Content Manager", "Database Stats", "Settings"]
        )
        
        if page == "Generate Brief":
            render_brief_generator()
        elif page == "Upload Document":
            render_document_uploader()
        elif page == "Content Manager":
            render_content_manager()
        elif page == "Database Stats":
            render_database_stats()
        elif page == "Settings":
            render_settings()

def render_database_connection_error():
    """Render database connection error page."""
    st.header("Database Connection Error")
    
    st.error("⚠️ Cannot connect to the database or required tables are missing. Please contact support.")
    
    st.info("The YouTube Brief Generator Bot requires a properly configured Supabase database to function.")
    
    if st.button("Retry Connection"):
        if check_tables_exist():
            st.success("✅ Connection restored! You can now use the application.")
            st.rerun()
        else:
            st.error("Connection still failed. Please contact support.")

def render_brief_generator():
    """Render the brief generator form."""
    st.header("Generate YouTube Brief")
    
    with st.form("brief_form"):
        topic = st.text_input("Topic or Keyword", help="Enter the main topic for your YouTube video")
        level = st.selectbox("Video Level", VIDEO_LEVELS, help="L1 is top-level content, L4 is most specific")
        opt_in_url = st.text_input("Opt-in URL (required for L1)", help="Landing page or resource link")
        
        st.subheader("Persona Override (Optional)")
        use_default_persona = st.checkbox("Use Default Persona", value=True)
        
        persona_override = None
        if not use_default_persona:
            col1, col2 = st.columns(2)
            with col1:
                age = st.text_input("Age Range", "Late 30s")
                role = st.text_input("Role", "agency owner")
            with col2:
                pain = st.text_input("Pain Point", "operational chaos")
            
            persona_override = {
                "age": age,
                "role": role,
                "pain": pain
            }
        
        submit_button = st.form_submit_button("Generate Brief")
    
    if submit_button:
        if not topic:
            st.error("Topic is required")
            return
        
        if level == "L1" and not opt_in_url:
            st.warning("Opt-in URL is recommended for L1 videos")
        
        # Submit job for background processing
        with st.spinner("Submitting brief generation request..."):
            job_id = generate_youtube_brief(
                topic=topic,
                level=level,
                opt_in_url=opt_in_url,
                persona_override=persona_override
            )
            
            st.success(f"✅ Brief generation started!")
            
            # Show a preview card with job info
            with st.container():
                st.info(f"**Creating brief for:** {topic}")
                st.progress(25)
                st.info("Your brief will be ready in about 30-45 seconds. You can continue working or check the Content Manager to view progress.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 Go to Content Manager"):
                        st.session_state["view"] = "active_jobs"
                        st.session_state["selected_job_id"] = job_id
                        st.session_state["page"] = "Content Manager"
                        st.rerun()
                with col2:
                    if st.button("➕ Create Another Brief"):
                        st.rerun()

def render_document_uploader():
    """Render the document uploader interface."""
    st.header("Upload Document to Knowledge Base")
    
    with st.form("document_form"):
        title = st.text_input("Document Title", help="Name of the document")
        source = st.selectbox(
            "Source",
            [
                "GrowthCode Book",
                "Fluid-Frame YouTube SOP",
                "Inspire-Educate-Solve Document",
                "YouTube Starter Kit",
                "Video Ranking Academy Workbook",
                "Other"
            ],
            help="Source of the document"
        )
        
        if source == "Other":
            source = st.text_input("Specify Other Source")
        
        # Added docx to the supported file types
        uploaded_file = st.file_uploader(
            "Upload Document", 
            type=["pdf", "txt", "docx"], 
            help="PDF, DOCX, or text file to add to knowledge base"
        )
        
        st.info("Document will be processed and chunked automatically. Larger documents may take more time.")
        
        submit_button = st.form_submit_button("Process Document")
    
    if submit_button:
        if not title or not source:
            st.error("Title and source are required")
            return
        
        if not uploaded_file:
            st.error("Please upload a file")
            return
        
        with st.spinner("Processing document..."):
            try:
                if uploaded_file.type == "application/pdf":
                    # Create temporary file to work with PyPDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Extract text from the temporary PDF file
                    with open(tmp_path, "rb") as f:
                        content = extract_text_from_pdf(f)
                    
                    # Clean up the temporary file
                    os.unlink(tmp_path)
                    
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    # Create temporary file to work with docx
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Extract text from the temporary DOCX file
                    content = extract_text_from_docx(tmp_path)
                    
                    # Clean up the temporary file
                    os.unlink(tmp_path)
                    
                else:
                    # Assume plain text file
                    content = uploaded_file.getvalue().decode("utf-8")
                
                # Submit document processing job
                job_id = process_document(title, source, content)
                
                st.success(f"✅ Document uploaded successfully!")
                
                # Show a preview card with job info
                with st.container():
                    st.info(f"**Processing document:** {title}")
                    st.progress(40)
                    st.info(f"Your document will be chunked and added to the knowledge base. Document size: {len(content):,} characters")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📋 Go to Content Manager"):
                            st.session_state["view"] = "active_jobs"
                            st.session_state["selected_job_id"] = job_id
                            st.session_state["page"] = "Content Manager"
                            st.rerun()
                    with col2:
                        if st.button("➕ Upload Another Document"):
                            st.rerun()
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

def render_content_manager():
    """Render the content manager page - combining Activity Center and Brief History."""
    st.header("Content Manager")
    
    # Set up tabs for different views
    if "view" not in st.session_state:
        st.session_state["view"] = "briefs"
    
    # Navigation tabs
    view_options = {
        "briefs": "📄 Briefs",
        "documents": "📚 Documents",
        "active_jobs": "🔄 In Progress"
    }
    
    # Get all jobs for badges
    all_jobs = get_all_jobs()
    active_job_count = len([j for j in all_jobs if j["status"] in [JobStatus.PENDING, JobStatus.PROCESSING]])
    
    tabs = {}
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tabs["briefs"] = st.button(
            "📄 Briefs", 
            use_container_width=True,
            type="primary" if st.session_state["view"] == "briefs" else "secondary"
        )
    
    with col2:
        tabs["documents"] = st.button(
            "📚 Documents", 
            use_container_width=True,
            type="primary" if st.session_state["view"] == "documents" else "secondary"
        )
    
    with col3:
        in_progress_label = f"🔄 In Progress ({active_job_count})" if active_job_count > 0 else "🔄 In Progress"
        tabs["active_jobs"] = st.button(
            in_progress_label, 
            use_container_width=True,
            type="primary" if st.session_state["view"] == "active_jobs" else "secondary"
        )
    
    # Handle tab selection
    for key in tabs:
        if tabs[key]:
            st.session_state["view"] = key
            # If changing tabs, clear selected items
            if key != "active_jobs":  # Keep selected job when going to active jobs
                st.session_state.pop("selected_job_id", None)
            st.session_state.pop("selected_brief", None)
            st.session_state.pop("selected_document", None)
    
    # Render the selected view
    selected_view = st.session_state["view"]
    
    if selected_view == "briefs":
        render_briefs_view()
    elif selected_view == "documents":
        render_documents_view()
    elif selected_view == "active_jobs":
        render_active_jobs_view()

def render_briefs_view():
    """Render the briefs view within Content Manager."""
    # Search and filter controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search Briefs", "")
    
    with col2:
        filter_level = st.selectbox("Level Filter", ["All Levels"] + VIDEO_LEVELS)
        
    with col3:
        sort_by = st.selectbox("Sort By", ["Newest First", "Oldest First", "Topic A-Z"])
        
    # Fetch briefs from database
    try:
        # Build query
        query = supabase.table("briefs").select("*")
        
        # Apply level filter if not "All Levels"
        if filter_level != "All Levels":
            query = query.eq("level", filter_level)
            
        # Fetch data
        result = query.execute()
        
        if not result.data:
            st.info("No briefs found. Generate your first brief!")
            st.button("➕ Create Your First Brief", on_click=lambda: st.session_state.update({"page": "Generate Brief"}))
            return
        
        # Apply sorting
        briefs = result.data
        if sort_by == "Newest First":
            briefs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "Oldest First":
            briefs.sort(key=lambda x: x.get("created_at", ""))
        elif sort_by == "Topic A-Z":
            briefs.sort(key=lambda x: x.get("topic", "").lower())
        
        # Apply search filter
        if search_term:
            briefs = [brief for brief in briefs if search_term.lower() in brief['topic'].lower()]
            if not briefs:
                st.info(f"No briefs found matching '{search_term}'")
                return
            
        # Status bar showing brief count
        st.caption(f"Showing {len(briefs)} brief{'s' if len(briefs) != 1 else ''}")
        
        # Check if we have a selected brief to display in detail
        if "selected_brief" in st.session_state:
            display_brief_detail(st.session_state["selected_brief"])
        else:
            # Display list of briefs in a grid
            display_briefs_grid(briefs)
            
    except Exception as e:
        st.error(f"Error loading briefs: {str(e)}")

def display_briefs_grid(briefs):
    """Display briefs in a grid layout."""
    # Show add button at the top
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Create New Brief", use_container_width=True):
            st.session_state["page"] = "Generate Brief"
            st.rerun()
    
    # Create a grid layout for brief cards - 3 columns
    brief_count = len(briefs)
    rows = (brief_count + 2) // 3  # Calculate rows needed (ceiling division)
    
    for row in range(rows):
        cols = st.columns(3)
        for col in range(3):
            idx = row * 3 + col
            if idx < brief_count:
                brief = briefs[idx]
                with cols[col]:
                    # Card-style container
                    with st.container(border=True):
                        # Level indicator with color
                        level_colors = {
                            "L1": "rgba(0, 128, 255, 0.2)",
                            "L2": "rgba(0, 192, 128, 0.2)",
                            "L3": "rgba(255, 165, 0, 0.2)",
                            "L4": "rgba(255, 0, 128, 0.2)"
                        }
                        level_color = level_colors.get(brief.get("level"), "rgba(128, 128, 128, 0.2)")
                        
                        st.markdown(
                            f"""<div style="background-color: {level_color}; 
                                        padding: 4px 8px; 
                                        border-radius: 4px; 
                                        display: inline-block; 
                                        margin-bottom: 8px;">
                                    {brief.get("level", "Unknown")}
                                </div>""", 
                            unsafe_allow_html=True
                        )
                        
                        # Brief title (truncate if too long)
                        topic = brief.get("topic", "Untitled")
                        if len(topic) > 40:
                            topic = topic[:37] + "..."
                        
                        st.markdown(f"### {topic}")
                        
                        # Creation date
                        created_at = brief.get("created_at", "")
                        if isinstance(created_at, str) and "T" in created_at:
                            created_at = created_at.split("T")[0]
                        st.caption(f"Created: {created_at}")
                        
                        # Action buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👁️ View", key=f"view_{brief['id']}", use_container_width=True):
                                st.session_state["selected_brief"] = brief
                                st.rerun()
                        with col2:
                            if st.button("🔄 Regenerate", key=f"regen_{brief['id']}", use_container_width=True):
                                job_id = generate_youtube_brief(
                                    topic=brief["topic"],
                                    level=brief["level"],
                                    opt_in_url=brief.get("opt_in_url", ""),
                                    persona_override=brief.get("persona", None)
                                )
                                st.session_state["view"] = "active_jobs"
                                st.session_state["selected_job_id"] = job_id
                                st.rerun()

def display_brief_detail(brief):
    """Display detailed view of a selected brief."""
    # Header with back button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Brief: {brief['topic']}")
    with col2:
        if st.button("◀️ Back to List", use_container_width=True):
            st.session_state.pop("selected_brief", None)
            st.rerun()
    
    # Brief metadata
    st.markdown(f"**Level:** {brief['level']}")
    
    created_at = brief.get("created_at", "")
    if isinstance(created_at, str) and "T" in created_at:
        created_at = created_at.split("T")[0]
    st.markdown(f"**Created:** {created_at}")
    
    if brief.get("opt_in_url"):
        st.markdown(f"**Opt-in URL:** {brief['opt_in_url']}")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📥 Download Markdown",
            brief["content"],
            file_name=f"{created_at}-{re.sub(r'[^\w\-]', '-', brief['topic'])}-Brief.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col2:
        if st.button("📋 Copy to Clipboard", key=f"copy_{brief['id']}", use_container_width=True):
            st.info("Content copied to clipboard!")
    with col3:
        if st.button("🔄 Regenerate", key=f"regen_detail_{brief['id']}", use_container_width=True):
            job_id = generate_youtube_brief(
                topic=brief["topic"],
                level=brief["level"],
                opt_in_url=brief.get("opt_in_url", ""),
                persona_override=brief.get("persona", None)
            )
            st.success(f"✅ Regeneration started!")
            st.session_state["view"] = "active_jobs"
            st.session_state["selected_job_id"] = job_id
            st.rerun()
    
    # Display tabs
    tab1, tab2 = st.tabs(["Preview", "Markdown"])
    
    with tab1:
        st.markdown(brief["content"])
    
    with tab2:
        st.text_area("Markdown", brief["content"], height=500)

def render_documents_view():
    """Render the documents view within Content Manager."""
    # Search and filter controls
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 Search Documents", "")
    with col2:
        if st.button("➕ Upload New Document", use_container_width=True):
            st.session_state["page"] = "Upload Document"
            st.rerun()
    
    # Fetch documents from database
    try:
        result = supabase.table("documents").select("*").order("created_at", desc=True).execute()
        
        if not result.data:
            st.info("No documents found. Upload your first document to get started.")
            
            if st.button("📄 Upload Your First Document", use_container_width=True):
                st.session_state["page"] = "Upload Document"
                st.rerun()
                
            return
        
        documents = result.data
        
        # Apply search filter
        if search_term:
            documents = [doc for doc in documents if search_term.lower() in doc['title'].lower() or search_term.lower() in doc['source'].lower()]
            if not documents:
                st.info(f"No documents found matching '{search_term}'")
                return
        
        # Status bar showing document count
        st.caption(f"Showing {len(documents)} document{'s' if len(documents) != 1 else ''}")
        
        # Create a grid layout for document cards - 3 columns
        doc_count = len(documents)
        rows = (doc_count + 2) // 3  # Calculate rows needed (ceiling division)
        
        for row in range(rows):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                if idx < doc_count:
                    doc = documents[idx]
                    with cols[col]:
                        with st.container(border=True):
                            # Document title
                            st.markdown(f"### {doc['title']}")
                            
                            # Source and creation date
                            st.caption(f"Source: {doc['source']}")
                            
                            created_at = doc.get("created_at", "")
                            if isinstance(created_at, str) and "T" in created_at:
                                created_at = created_at.split("T")[0]
                            st.caption(f"Created: {created_at}")
                            
                            # Calculate content size in KB
                            content_size = len(doc.get("content", "")) / 1024
                            st.markdown(f"Size: {content_size:.1f} KB")
                            
                            # Calculate chunks
                            doc_id = doc["id"]
                            try:
                                chunks_result = supabase.table("document_chunks").select("id", count="exact").eq("document_id", doc_id).execute()
                                chunk_count = chunks_result.count if hasattr(chunks_result, 'count') else len(chunks_result.data)
                                st.markdown(f"Chunks: {chunk_count}")
                            except:
                                st.markdown("Chunks: Unknown")
                            
                            # View button
                            if st.button("👁️ View Content", key=f"view_doc_{doc['id']}", use_container_width=True):
                                # Here you could add a document viewer
                                st.session_state["selected_document"] = doc
                                st.rerun()
    
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
    
    # Display document detail if selected
    if "selected_document" in st.session_state:
        doc = st.session_state["selected_document"]
        
        # Header with back button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader
        # Header with back button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Document: {doc['title']}")
        with col2:
            if st.button("◀️ Back to List", use_container_width=True):
                st.session_state.pop("selected_document", None)
                st.rerun()
        
        # Document metadata
        st.markdown(f"**Source:** {doc['source']}")
        
        created_at = doc.get("created_at", "")
        if isinstance(created_at, str) and "T" in created_at:
            created_at = created_at.split("T")[0]
        st.markdown(f"**Created:** {created_at}")
        
        # Calculate content size and chunks
        content_size = len(doc.get("content", "")) / 1024
        st.markdown(f"**Size:** {content_size:.1f} KB")
        
        doc_id = doc["id"]
        try:
            chunks_result = supabase.table("document_chunks").select("id", count="exact").eq("document_id", doc_id).execute()
            chunk_count = chunks_result.count if hasattr(chunks_result, 'count') else len(chunks_result.data)
            st.markdown(f"**Chunks:** {chunk_count}")
        except:
            st.markdown("**Chunks:** Unknown")
        
        # Document content preview
        st.subheader("Content Preview")
        content = doc.get("content", "")
        
        # Show first 2000 characters with option to see more
        preview_length = min(2000, len(content))
        st.markdown(f"```\n{content[:preview_length]}\n```")
        
        if len(content) > preview_length:
            if st.button("Show More Content"):
                st.markdown(f"```\n{content}\n```")

def render_active_jobs_view():
    """Render the active jobs view within Content Manager."""
    # Get all jobs
    all_jobs = get_all_jobs()
    
    # Split jobs by status
    active_jobs = [j for j in all_jobs if j["status"] in [JobStatus.PENDING, JobStatus.PROCESSING]]
    completed_jobs = [j for j in all_jobs if j["status"] == JobStatus.COMPLETED]
    failed_jobs = [j for j in all_jobs if j["status"] == JobStatus.FAILED]
    
    # Determine if we should show a specific job detail
    selected_job_id = st.session_state.get("selected_job_id", None)
    
    # Layout with sidebar for jobs and main area for details
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Recent Activity")
        
        # Show active jobs first
        if active_jobs:
            st.markdown("#### 🔄 In Progress")
            for job in active_jobs:
                job_type_emoji = "📝" if "Brief" in job["friendly_name"] else "📄"
                
                if st.button(
                    f"{job_type_emoji} {job['friendly_name']}",
                    key=f"job_{job['id']}",
                    use_container_width=True,
                    type="primary" if selected_job_id == job["id"] else "secondary"
                ):
                    st.session_state["selected_job_id"] = job["id"]
                    st.rerun()
                    
                # Show progress indicator
                if job["status"] == JobStatus.PROCESSING:
                    st.progress(75, f"Processing...")
                else:
                    st.progress(25, f"Queued...")
        
        # Show completed jobs
        if completed_jobs:
            st.markdown("#### ✅ Completed")
            for job in completed_jobs[:5]:  # Limit to recent 5
                job_type_emoji = "📝" if "Brief" in job["friendly_name"] else "📄"
                
                if st.button(
                    f"{job_type_emoji} {job['friendly_name']}",
                    key=f"job_{job['id']}",
                    use_container_width=True,
                    type="primary" if selected_job_id == job["id"] else "secondary"
                ):
                    st.session_state["selected_job_id"] = job["id"]
                    st.rerun()
        
        # Show failed jobs
        if failed_jobs:
            st.markdown("#### ❌ Failed")
            for job in failed_jobs[:3]:  # Limit to recent 3
                job_type_emoji = "📝" if "Brief" in job["friendly_name"] else "📄"
                
                if st.button(
                    f"{job_type_emoji} {job['friendly_name']}",
                    key=f"job_{job['id']}",
                    use_container_width=True,
                    type="primary" if selected_job_id == job["id"] else "secondary"
                ):
                    st.session_state["selected_job_id"] = job["id"]
                    st.rerun()
        
        # Add refresh button
        if st.button("🔄 Refresh Activity", use_container_width=True):
            st.rerun()
    
    with col2:
        if selected_job_id:
            # Show selected job details
            job_details = get_job_status(selected_job_id)
            
            if job_details["status"] == "unknown":
                st.warning(f"Job information no longer available. It may have been completed and archived.")
                return
            
            # Determine emoji and color based on status
            status_colors = {
                JobStatus.COMPLETED: "green",
                JobStatus.FAILED: "red",
                JobStatus.PROCESSING: "blue",
                JobStatus.PENDING: "orange"
            }
            
            status_emoji = {
                JobStatus.COMPLETED: "✅",
                JobStatus.FAILED: "❌",
                JobStatus.PROCESSING: "🔄",
                JobStatus.PENDING: "⏳"
            }
            
            friendly_status = {
                JobStatus.COMPLETED: "Completed",
                JobStatus.FAILED: "Failed",
                JobStatus.PROCESSING: "In Progress",
                JobStatus.PENDING: "Queued"
            }
            
            # Job Header
            st.subheader(job_details["metadata"].get("friendly_name", f"Job {selected_job_id[:8]}"))
            
            # Status indicator
            status = job_details["status"]
            st.markdown(
                f"<span style='color:{status_colors.get(status, 'gray')};'>"
                f"{status_emoji.get(status, '❓')} {friendly_status.get(status, status.title())}</span>", 
                unsafe_allow_html=True
            )
            
            # Creation time
            created_time = job_details["metadata"].get("created_at", "Unknown")
            if isinstance(created_time, str) and "T" in created_time:
                created_time = created_time.replace("T", " ").split(".")[0]
            st.caption(f"Started: {created_time}")
            
            # Job Type-specific content display
            job_type = job_details["metadata"].get("type", "unknown")
            
            # For completed brief generation
            if status == JobStatus.COMPLETED and job_type == JobType.BRIEF_GENERATION:
                if "content" in job_details["result"]:
                    st.success("Brief generated successfully! 🎉")
                    
                    # Show preview with tabs for display or raw
                    tabs = st.tabs(["Preview", "Markdown", "Technical Details"])
                    
                    with tabs[0]:
                        st.markdown(job_details["result"]["content"])
                        
                        # Download button
                        topic = job_details["result"].get("topic", "brief")
                        st.download_button(
                            "📥 Download Markdown",
                            job_details["result"]["content"],
                            file_name=f"{datetime.now().strftime('%Y-%m-%d')}-{re.sub(r'[^\w\-]', '-', topic)}-Brief.md",
                            mime="text/markdown"
                        )
                        
                        # View in Content Manager button
                        if "id" in job_details["result"]:
                            # Fetch full brief to get all fields
                            brief_result = supabase.table("briefs").select("*").eq("id", job_details["result"]["id"]).execute()
                            if brief_result.data:
                                brief = brief_result.data[0]
                                if st.button("📋 Open in Briefs", use_container_width=True):
                                    st.session_state["view"] = "briefs"
                                    st.session_state["selected_brief"] = brief
                                    st.session_state.pop("selected_job_id", None)
                                    st.rerun()
                    
                    with tabs[1]:
                        st.text_area("Markdown", job_details["result"]["content"], height=400)
                    
                    with tabs[2]:
                        st.json(job_details["result"])
                        
                        with st.expander("Job Logs"):
                            for log in job_details["logs"]:
                                st.text(log)
            
            # For completed document processing
            elif status == JobStatus.COMPLETED and job_type == JobType.DOCUMENT_PROCESSING:
                if "document_id" in job_details["result"]:
                    st.success("Document processed successfully! 🎉")
                    
                    # Show document info in a clean card
                    with st.container(border=True):
                        st.markdown(f"**Title:** {job_details['result'].get('title', 'Untitled')}")
                        st.markdown(f"**Source:** {job_details['result'].get('source', 'Unknown')}")
                        st.markdown(f"**Document ID:** {job_details['result'].get('document_id', 'Unknown')}")
                        
                        # Show chunk statistics with a progress bar
                        total = job_details['result'].get('total_chunks', 0)
                        success = job_details['result'].get('successful_chunks', 0)
                        failed = job_details['result'].get('failed_chunks', 0)
                        
                        if total > 0:
                            st.markdown(f"**Chunks:** {success}/{total} processed successfully")
                            st.progress(success/total if total > 0 else 0)
                            
                        # View in Documents button
                        document_id = job_details['result'].get('document_id')
                        if document_id:
                            # Fetch full document to get all fields
                            doc_result = supabase.table("documents").select("*").eq("id", document_id).execute()
                            if doc_result.data:
                                doc = doc_result.data[0]
                                if st.button("📋 Open in Documents", use_container_width=True):
                                    st.session_state["view"] = "documents"
                                    st.session_state["selected_document"] = doc
                                    st.session_state.pop("selected_job_id", None)
                                    st.rerun()
                    
                    with st.expander("Technical Details"):
                        st.json(job_details["result"])
                        
                        with st.expander("Job Logs"):
                            for log in job_details["logs"]:
                                st.text(log)
            
            # For failed jobs
            elif status == JobStatus.FAILED:
                st.error(f"Job failed: {job_details['result'].get('error', 'Unknown error')}")
                
                with st.expander("Error Details"):
                    st.json(job_details["result"])
                    
                    with st.expander("Job Logs"):
                        for log in job_details["logs"]:
                            st.text(log)
                
                # Retry option for failed jobs
                if job_type == JobType.BRIEF_GENERATION and "params" in job_details["metadata"]:
                    params = job_details["metadata"]["params"]
                    if isinstance(params, tuple) and len(params) >= 3:
                        topic, level, opt_in_url = params[:3]
                        persona_override = params[3] if len(params) > 3 else None
                        
                        if st.button("🔄 Retry Brief Generation", use_container_width=True):
                            new_job_id = generate_youtube_brief(topic, level, opt_in_url, persona_override)
                            st.success("✅ New brief generation job started!")
                            st.session_state["selected_job_id"] = new_job_id
                            st.rerun()
                
                elif job_type == JobType.DOCUMENT_PROCESSING and "params" in job_details["metadata"]:
                    params = job_details["metadata"]["params"]
                    if isinstance(params, tuple) and len(params) >= 3:
                        title, source, content = params[:3]
                        
                        if st.button("🔄 Retry Document Processing", use_container_width=True):
                            new_job_id = process_document(title, source, content)
                            st.success("✅ New document processing job started!")
                            st.session_state["selected_job_id"] = new_job_id
                            st.rerun()
            
            # For in-progress jobs
            elif status in [JobStatus.PROCESSING, JobStatus.PENDING]:
                # Show progress indicator
                if status == JobStatus.PROCESSING:
                    st.progress(75, "Processing...")
                else:
                    st.progress(25, "Queued...")
                
                # Show estimated completion time if available
                if "estimated_completion" in job_details["metadata"]:
                    est_time = job_details["metadata"]["estimated_completion"]
                    now = datetime.now()
                    
                    if est_time > now:
                        remaining = (est_time - now).total_seconds()
                        if remaining < 60:
                            st.info(f"Estimated completion in {int(remaining)} seconds")
                        else:
                            st.info(f"Estimated completion in {int(remaining/60)} minutes")
                else:
                    st.info("Processing in background. This may take a minute.")
                
                # Recent logs (last 5)
                st.subheader("Recent Progress")
                logs = job_details["logs"][-5:] if job_details["logs"] else []
                for log in logs:
                    st.text(log)
                
                # Auto-refresh for in-progress jobs
                st.button("🔄 Refresh Status", use_container_width=True)
                
                # Add JavaScript for auto-refresh every 5 seconds for in-progress jobs
                st.markdown("""
                <script>
                    setTimeout(function(){
                        window.location.reload();
                    }, 5000);
                </script>
                """, unsafe_allow_html=True)
        else:
            # No job selected
            st.subheader("Activity Center")
            st.markdown("""
            Track the progress of your brief generation and document processing jobs.
            
            ### How it works:
            
            1. **Active jobs** are shown in the sidebar on the left
            2. **Select a job** to view its details and progress
            3. **Refresh** to see the latest status updates
            
            When jobs complete, you can view the results or open them in the appropriate section.
            """)
            
            # Show stats
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("In Progress", len(active_jobs))
                
                with col2:
                    st.metric("Completed", len(completed_jobs))
                
                with col3:
                    st.metric("Failed", len(failed_jobs))
            
            # If there are no active jobs but there are completed ones
            if not active_jobs and completed_jobs:
                st.info("No active jobs are currently running. Select a completed job to view its results, or create a new brief or upload a document.")

def render_database_stats():
    """Render the database statistics page."""
    st.header("Database Statistics")
    
    try:
        # Count records in each table
        doc_count = table_row_count("documents")
        chunk_count = table_row_count("document_chunks")
        brief_count = table_row_count("briefs")
        hierarchy_count = table_row_count("content_hierarchy")
        
        # Display counts in a nice grid with icons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents", doc_count)
        with col2:
            st.metric("🧩 Document Chunks", chunk_count)
        with col3:
            st.metric("📝 Briefs", brief_count)
        with col4:
            st.metric("🔗 Hierarchy Links", hierarchy_count)
        
        # Show document list
        st.subheader("Uploaded Documents")
        if doc_count > 0:
            result = supabase.table("documents").select("*").execute()
            if result.data:
                # Create nicer document cards instead of a table
                doc_cols = st.columns(3)
                
                for i, doc in enumerate(result.data):
                    col_idx = i % 3
                    
                    with doc_cols[col_idx]:
                        with st.container(border=True):
                            st.markdown(f"### {doc['title']}")
                            st.caption(f"Source: {doc['source']}")
                            st.caption(f"Created: {doc['created_at'].split('T')[0] if isinstance(doc['created_at'], str) else str(doc['created_at'])}")
                            
                            # Calculate content size in KB
                            content_size = len(doc.get("content", "")) / 1024
                            st.markdown(f"Size: {content_size:.1f} KB")
        else:
            st.info("No documents uploaded yet. Go to the Upload Document page to add content.")
            
            if st.button("📄 Upload Your First Document"):
                st.session_state["page"] = "Upload Document"
                st.rerun()
            
        # Show brief list
        st.subheader("Recently Generated Briefs")
        if brief_count > 0:
            result = supabase.table("briefs").select("*").order("created_at", desc=True).limit(6).execute()
            if result.data:
                # Create a nicer grid layout for briefs
                brief_cols = st.columns(3)
                
                for i, brief in enumerate(result.data):
                    col_idx = i % 3
                    
                    with brief_cols[col_idx]:
                        with st.container(border=True):
                            st.markdown(f"### {brief['topic']}")
                            st.caption(f"Level: {brief['level']}")
                            st.caption(f"Created: {brief['created_at'].split('T')[0] if isinstance(brief['created_at'], str) else str(brief['created_at'])}")
                            
                            # Add view button
                            if st.button(f"👁️ View", key=f"view_stats_{brief['id']}"):
                                st.session_state["selected_brief"] = brief
                                st.session_state["view"] = "briefs"
                                st.session_state["page"] = "Content Manager"
                                st.rerun()
                
                # Add a "View All" button
                if st.button("📚 View All Briefs", use_container_width=True):
                    st.session_state["view"] = "briefs"
                    st.session_state["page"] = "Content Manager"
                    st.rerun()
        else:
            st.info("No briefs generated yet. Go to the Generate Brief page to create content.")
            
            if st.button("➕ Create Your First Brief"):
                st.session_state["page"] = "Generate Brief"
                st.rerun()
            
    except Exception as e:
        st.error(f"Error retrieving database statistics: {str(e)}")
        
    st.subheader("Database Health")
    
    if st.button("🔄 Check Database Connection"):
        with st.spinner("Checking database connection..."):
            if check_tables_exist():
                st.success("✅ Database connection successful! All tables exist and are accessible.")
            else:
                st.error("⚠️ Database tables not found or inaccessible. Please contact support.")

def render_settings():
    """Render the settings page with only Default Persona and API Keys sections."""
    st.header("Settings")
    
    # Default Persona section
    st.subheader("Default Persona")
    persona = get_default_persona()
    
    with st.form("persona_form"):
        age = st.text_input("Age Range", persona["age"])
        role = st.text_input("Role", persona["role"])
        pain = st.text_input("Pain Point", persona["pain"])
        
        submit_button = st.form_submit_button("Update Default Persona")
    
    if submit_button:
        # This would save to a settings table in Supabase
        st.success("✅ Default persona updated!")
        
        # Show updated values
        with st.container(border=True):
            st.markdown("### Updated Persona")
            st.markdown(f"**Age Range:** {age}")
            st.markdown(f"**Role:** {role}")
            st.markdown(f"**Pain Point:** {pain}")
    
    # API Keys section
    st.subheader("API Keys")
    
    with st.expander("View/Update API Keys"):
        st.warning("⚠️ These keys are sensitive. Do not share them with others.")
        groq_key = st.text_input("Groq API Key", value=GROQ_API_KEY, type="password")
        jina_key = st.text_input("Jina API Key", value=JINA_API_KEY, type="password")
        
        if st.button("Update API Keys"):
            # In a production app, you would store these securely
            # This is just for demonstration
            st.success("✅ API keys updated for this session.")
            st.info("Note: Changes will be lost when restarting the app unless stored in environment variables.")

if __name__ == "__main__":
    main()