# Runnables
Runnables in LangChain are modular units of computation that provide a standardized interface for executing tasks, allowing components like LLMs, prompts, and parsers to be chained together seamlessly.  
They serve as the building blocks for LangChain Expression Language (LCEL), enabling developers to build complex, scalable AI workflows by connecting these components like building blocks.

In LangChain, Runnables are the building blocks that represent any unit of work (like calling an LLM, formatting a prompt, or transforming data). They follow a common interface (invoke, batch, stream), which makes them composable.

## 1. Task Specific Runnables
These are core LangChain components that have been converted into Runnables so they can be used in pipelines.
They are specialized runnables designed for specific tasks, often wrapping more complex logic internally.

### Examples:
* LLM Runnable → calls an LLM
* ChatModel Runnable → chat-based interaction
* Retriever Runnable → fetch documents
* Tool Runnable → execute tools
* Agent Runnable → decision-making workflows

### Key idea:

* 👉 They are like pre-built machines—optimized for a particular job.

### Example code: 
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
response = model.invoke("Explain LangChain")
```

* Here, the model itself is a runnable.

## 2. Runnables Primitives
These are the basic, generic units that do simple operations. They’re not tied to any specific task—they just process input → output.

### Examples:
* RunnableLambda → wrap a Python function
* RunnableMap → run multiple runnables in parallel
* RunnableSequence → chain steps together
* RunnablePassthrough → pass input unchanged

### Key idea:

* 👉 They are like LEGO pieces—small, flexible, and reusable.

### Example code
```python
from langchain_core.runnables import RunnableSequence

chain = RunnableSequence(
    RunnableLambda(lambda x: x + " world"),
    RunnableLambda(lambda x: x.upper())
)

print(chain.invoke("hello"))  # HELLO WORLD
```