def convert_to_documents(dataset):
    """Convert Hugging Face dataset to LangChain documents"""
    documents = []

    for i, example in enumerate(dataset):

        question = example.get("question", "")
        context = example.get("context", "")
        long_answer = example.get("long_answer", "")
        final_decision = example.get("final_decision", "")

        # Combine the information into a single text
        content = f"""Question: {question}

Context: {context}

Answer: {long_answer}

Final Decision: {final_decision}"""

        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "question": question,
                "final_decision": final_decision,
                "source": "pubmed_qa",
                "index": i,
            },
        )
        documents.append(doc)

        # Limit to first 10000 documents for performance
        if i >= 2000:
            break

    return documents


# print("Converting dataset to documents...")
# docs = convert_to_documents(dataset)

# Split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(docs)

# print(f"Created {len(texts)} text chunks from the dataset")