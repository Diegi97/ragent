# RAGent

RAGent is a project focused on training intelligent agents that can navigate and extract information from diverse knowledge bases. Whether it's an Elasticsearch index, vector database, traditional file system, SQL database, or any combination of these—our agent learns to work with them all.

## What We're Building

We're training our RAG agent using Reinforcement Learning from Verifiable Rewards, leveraging LLM-based reward functions to guide learning. The training infrastructure is built on the verifiers library, and we're generating synthetic datasets of question-answer pairs along with agent interaction trajectories to power the learning process.

Think of it as creating an agent that can perform sophisticated multi-hop question answering (similar to o3) across any search system configuration, adapting to different data sources and tool combinations on the fly.

## How It Works: Environments and Data Generation

Our training approach centers around modular environments, each consisting of three key components:

**Search Engine Layer**  
The backbone of each environment—whether it's Elasticsearch, BM25s, Qdrant, or hybrid combinations that leverage the strengths of multiple systems.

**Tool Interface**  
The agent's way of interacting with the knowledge base. This might include `search(query: str)` for searching, `read(doc_id: int)` for document retrieval, file-system navigation tools, or SQL console access.

**Data Source**  
The actual knowledge repository, such as developer documentation, email archives, or internal company documents. These sources also serve as the foundation for generating our training QA pairs.

### Environment Examples

- **Hybrid Search Setup**: Elasticsearch + Qdrant combination with `search` and `read` tools
- **Lightweight Lexical Search**: BM25s over developer documentation for fast local development
- **Interactive Systems**: File system navigation or database query interfaces

Since deploying heavy infrastructure like Elasticsearch can be resource-intensive, we prioritize lightweight local alternatives (like BM25s) that capture the essential behaviors while remaining accessible for development and experimentation.

## Data Generation Strategy

Our data generation pipeline starts with the WebShaper approach, which can function as both a data source and an environment itself. We're intentionally focusing on non-web data sources—email archives, developer documentation, internal company documents—since existing research tools already provide comprehensive web coverage.

The workflow involves mixing and matching data sources with different search engines and tool combinations, generating QA pairs, validating their quality, and packaging everything into reusable training environments.

## Current Development

We're beginning with lexical search over general developer documentation as our first environment. Once we have the QA generation pipeline running smoothly and several search systems operational, we'll expand to create a diverse set of validated environments ready for agent training.

The ultimate goal is an agent that generalizes across search paradigms and can tackle complex, multi-step information retrieval tasks regardless of the underlying knowledge base architecture.

## TODOs

- [ ] The `hf_dataset` string` argument passed to the environment's `load_environment` func should be a class itself containing the information needed to load the dataset (fd, local, etc...)
- [ ] Split ragagent_core (and potentially rename it) & environments into truly independent porjects