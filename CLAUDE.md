# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Philosophy

**Keep implementations minimal and straightforward. Avoid over-engineering.**

- Prefer simple, direct solutions over complex abstractions
- Choose readability over cleverness 
- Implement only what's needed, when it's needed
- Avoid premature optimization and unnecessary features
- Use standard library and well-established patterns

## Git Commit Guidelines

- Write clear, concise commit messages focused on the change
- Never mention Claude Code, Claude, or AI assistance in commit messages
- Focus on what was changed and why, not how it was created
- Use conventional commit format when appropriate

## Project Overview

This is a Warhammer 40k RAG (Retrieval-Augmented Generation) application that scrapes content from the Warhammer 40k Fandom wiki and provides a Streamlit chat interface for querying lore. The project is designed for deployment on Streamlit Cloud.

## Milestones

- Fetching raw html for each wiki page and storing in sqlite database: Done
- Parsing HTML and extracting relevant content to store in sqlite database: Done
- Creating chunks from extracted content and storing in sqlite database: Done
- Generating embeddings from text chunks and storing in Qdrant vector database: Done

## Next Step 

- Implementing the query engine to handle user queries and retrieve relevant information from the database.