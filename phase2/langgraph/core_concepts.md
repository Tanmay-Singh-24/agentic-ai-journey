# LangGraph — Core Concepts Reference

> For someone who already knows LangChain (chains, prompts, retrievers, tools, agents).
> Snippets verified against **LangGraph 1.2.2**. This is a lookup sheet, not a tutorial.

**Mental model:** LangChain's LCEL (`a | b | c`) is a *straight pipe* — data flows one way, start to finish. LangGraph is a *flowchart* — nodes can branch, loop back, and share a running **state**. You reach for LangGraph the moment you need cycles (agent loops), branching logic, or built-in memory.

```bash
pip install langgraph      # already installed: 1.2.2
```

---

## 1. What is a Graph

A graph is a set of **nodes** (units of work) connected by **edges** (what runs next). Unlike a chain, which is strictly linear, a graph can branch (go different ways based on state) and **loop** (revisit a node) — which is exactly what an agent's "think → act → think again" cycle needs. You describe *what connects to what*; LangGraph handles execution order.

```python
from langgraph.graph import StateGraph, START, END
# A graph is built around a State type (below), then nodes + edges are added to it.
graph = StateGraph(State)
```

---

## 2. State

**State** is a single shared dict that every node reads from and writes to — it's the data that flows through the graph (replacing the values you used to thread manually between chain steps). You define its shape with a `TypedDict` (or Pydantic model) so keys and types are explicit. Each node returns a **partial** dict; LangGraph merges it into the running state.

By default a returned key **overwrites** the old value. To instead **append/merge**, annotate the field with a **reducer** — e.g. `add_messages` accumulates a message list instead of replacing it (this is what makes chat history "just work").

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    question: str                                  # overwritten on each write (default)
    messages: Annotated[list, add_messages]        # reducer → new messages are APPENDED
```

---

## 3. Nodes

A **node** is just a Python function. It receives the **whole current state** and returns a **dict of only the keys it wants to change** (not the full state). The returned dict is merged in per the reducer rules above. Nodes are where your real work happens — call an LLM, run a retriever, execute a tool.

```python
def answer_node(state: State) -> dict:
    reply = llm.invoke(state["question"])          # read from state
    return {"messages": [reply]}                   # return ONLY what changed
```

---

## 4. Edges

**Edges** define routing. A **normal edge** is unconditional — after node A, always go to node B. A **conditional edge** runs a router function that reads the state and **returns the name of the next node** (or `END`), letting the graph branch or loop. The third arg maps the router's return values to actual node names.

```python
# normal edge: A always → B
graph.add_edge("retrieve", "answer")

# conditional edge: a function decides where to go next
def route(state: State) -> str:
    return "use_tool" if state["needs_tool"] else END

graph.add_conditional_edges(
    "answer",                                   # from this node
    route,                                      # router reads state, returns a key
    {"use_tool": "use_tool", END: END},         # key → destination node
)
```

---

## 5. StateGraph

`StateGraph` is the builder class. You instantiate it with your State type, register nodes with `add_node(name, fn)`, wire them with `add_edge` / `add_conditional_edges`, and mark entry/exit with `START` / `END` (Section 8).

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)        # name → function
graph.add_node("answer", answer_node)
graph.add_edge(START, "retrieve")                # entry point
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)                    # exit point
```

---

## 6. Compiling and Running

`.compile()` turns the builder into a runnable app (validates the wiring). The compiled app exposes the **same LangChain Runnable interface** you already know — `.invoke()`, `.stream()`, `.batch()` — so it drops into existing code. `.invoke(initial_state)` runs the graph from `START` to `END` and returns the final state dict.

```python
app = graph.compile()                            # build the executable graph
result = app.invoke({"question": "hi", "messages": []})
print(result["messages"])                        # final merged state
# .stream(...) yields state after each node — useful for watching an agent think
```

---

## 7. Checkpointer / Persistence (native memory)

A **checkpointer** saves a snapshot of the state after every step, keyed by a `thread_id` you pass at call time. Pass one to `.compile(checkpointer=...)` and memory becomes **automatic**: every invoke under the same `thread_id` resumes from the saved state — no manual history-passing. This is LangGraph's built-in replacement for LangChain's `RunnableWithMessageHistory` + `InMemoryChatMessageHistory` wiring. It also enables pause/resume and time-travel debugging.

```python
from langgraph.checkpoint.memory import InMemorySaver   # MemorySaver is an alias

app = graph.compile(checkpointer=InMemorySaver())       # swap for SqliteSaver/Postgres in prod

cfg = {"configurable": {"thread_id": "user-123"}}       # identifies the conversation
app.invoke({"messages": [("user", "My name is Tanmay")]}, config=cfg)
app.invoke({"messages": [("user", "What's my name?")]}, config=cfg)  # remembers — same thread

app.get_state(cfg).values        # inspect the persisted state for that thread
```

---

## 8. START and END

`START` and `END` are special sentinel nodes (not functions you write). An edge **from `START`** marks the graph's entry point — the first real node to run. An edge **to `END`** marks an exit — when execution reaches it, the graph stops and returns the state. A graph can have multiple edges into `END` (different branches finishing).

```python
from langgraph.graph import START, END

graph.add_edge(START, "first_node")    # where execution begins
graph.add_edge("last_node", END)       # where it stops
# Conditional edges can also route straight to END (see Section 4).
```

---

## LangChain → LangGraph Cheat Sheet

| LangChain concept | LangGraph equivalent | Note |
|---|---|---|
| LCEL chain (`a \| b \| c`) | A compiled `StateGraph` | Linear pipe → branching/looping flowchart |
| Runnable step / lambda | A **node** (Python function) | Node returns a partial state dict |
| Data passed implicitly between steps | **State** (`TypedDict`) | Explicit, shared, typed |
| `RunnablePassthrough` / manual dict plumbing | State keys + **reducers** | `add_messages` appends instead of overwriting |
| Fixed `\|` ordering | **Edges** (normal + conditional) | Routing can depend on state; can loop |
| `RunnableWithMessageHistory` + `ChatMessageHistory` | **Checkpointer** + `thread_id` | Native, automatic, persistent memory |
| `RunnableBranch` / if-else routing | **Conditional edges** | Router function returns next node name |
| `AgentExecutor` (the agent loop) | A graph with a node→tool→node **cycle** | You can see/control every step |
| `.invoke()` / `.stream()` | `.invoke()` / `.stream()` | Same Runnable interface after `.compile()` |
| (no real equivalent) | `START` / `END` sentinels | Explicit entry/exit markers |

**One-line summary:** chains run a fixed line; graphs run a stateful flowchart that can branch and loop — which is why every serious agent framework is built on graphs, not chains.
