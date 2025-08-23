# RAGent

The goal of this project is to train a RAG agent. This RAG agent should be able to navigate any knowledge base, be it a elasticsearch index, a vector database, a simple file system, SQL database, etc, or any combination of these.

This agent will be trained with Reinforcement Learning from Verifiable Rewards (sort of, because we will use a lot of LLM as a judge for the rewards function). We will use the verifiers library as a general library for the training. We also need to generate a synthetic dataset of questions and answers pairs to train the agent, and maybe another dataset of agent trajectories.