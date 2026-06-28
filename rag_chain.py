from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(llm):

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Use ONLY the context below to answer the question.

If the context contains relevant information,
provide a clear answer.

Only say
"Information not found in uploaded documents."
when the answer truly does not exist in the context.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = prompt | llm | StrOutputParser()

    return chain