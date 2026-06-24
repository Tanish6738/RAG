from pydantic import BaseModel, Field
from typing import Optional


class PageContent(BaseModel):
    """
    Represents the extracted text and structure of a single PDF page.
    """
    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Extracted raw text content of the page")
    tables: list[list[list[str]]] = Field(
        default_factory=list,
        description="Extracted table structures represented as table -> row -> cells"
    )
    has_images: bool = Field(False, description="Flag indicating if the page contains images")
    char_count: int = Field(..., description="Character length of the text")


class DocumentContent(BaseModel):
    """
    Represents a fully extracted PDF document.
    """
    filename: str = Field(..., description="Original filename of the PDF")
    total_pages: int = Field(..., description="Total pages in the PDF")
    pages: list[PageContent] = Field(..., description="List of individual page contents")
    metadata: dict[str, str] = Field(default_factory=dict, description="Metadata dictionary extracted from the PDF")
    extraction_method: str = Field(..., description="The method/library used, e.g., 'pypdf' or 'pdfplumber'")


class Chunk(BaseModel):
    """
    Represents a single text chunk that will be embedded and stored.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier formatted as '{filename}::p{page}::c{index}'")
    text: str = Field(..., description="Text content of the chunk")
    token_count: int = Field(..., description="Total token count for the chunk text")
    page_number: int = Field(..., description="Page number the chunk originated from")
    chunk_index: int = Field(..., description="Sequential position index within the document")
    source_filename: str = Field(..., description="Filename of the source document")
    metadata: dict = Field(default_factory=dict, description="Additional context metadata")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The search query or question to look up")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    filename_filter: Optional[str] = Field(None, description="Optional document filename to restrict the search")


class UsageInfo(BaseModel):
    prompt_tokens: int = Field(..., description="Tokens used for the prompt")
    completion_tokens: int = Field(..., description="Tokens generated for the answer")


class ChunkInfo(BaseModel):
    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    source_filename: str
    score: float


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer from the model")
    chunks: list[ChunkInfo] = Field(..., description="List of source chunks used for generating the answer")
    model: str = Field(..., description="LLM model used for generation")
    usage: Optional[UsageInfo] = Field(None, description="Model token usage details")


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="The user message to send to the agent")
    conversation_id: str = Field(..., description="UUID for conversation continuity")


class ToolTrace(BaseModel):
    tool: str = Field(..., description="The name of the tool called")
    args: dict = Field(..., description="The arguments passed to the tool")
    result_preview: str = Field("", description="A short preview of the tool execution result")


class AgentChatResponse(BaseModel):
    answer: str = Field(..., description="The final answer from the agent")
    conversation_id: str = Field(..., description="UUID for the conversation")
    tool_trace: list[ToolTrace] = Field(default_factory=list, description="Trace of all tools called during this turn")
    iterations: int = Field(..., description="Number of execution iterations taken by the agent loop")
    model: str = Field(..., description="LLM model used for the execution")
