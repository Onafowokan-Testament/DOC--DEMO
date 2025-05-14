

def convert_to_documents(dataset):
    """Convert Hugging Face dataset to LangChain documents"""
    documents = []

    # Verify dataset structure
    print(f"Dataset type: {type(dataset)}")
    print(f"First item type: {type(dataset[0])}")

    for i, example in enumerate(dataset):
        # Verify example type
        if not isinstance(example, dict):
            raise TypeError(f"Unexpected example type {type(example)} at index {i}")

        # Access fields using dictionary keys
        question = example.get("question", "")
        context = example.get("context", {})
        long_answer = example.get("long_answer", "")
        final_decision = example.get("final_decision", "")

        # Handle context which is itself a dictionary
        context_str = (
            context.get("contexts", [""])[0] if isinstance(context, dict) else ""
        )

        # Combine the information into a single text
        content = f"""Question: {question}
Context: {context_str}
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

        # Limit to first 3000 documents for performance
        if i >= 3000:
            break

    return documents
