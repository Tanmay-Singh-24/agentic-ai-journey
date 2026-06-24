"""
═══════════════════════════════════════════════════════════════════════════════
 LANGGRAPH AGENT FROM SCRATCH — the tool-calling loop, built as a graph
═══════════════════════════════════════════════════════════════════════════════

This is the payoff for tool_notes.py + core_concepts.md.

In phase2/tool_notes.py (Section 5) you HAND-WROTE this loop:

    while True:
        ai = llm.invoke(msgs)            # ask the model
        if not ai.tool_calls: break      # no tool wanted → done
        for call in ai.tool_calls:       # else run the tools
            run them, append ToolMessage
        # ...loop back and ask again

That while-loop IS an agent. Here we rebuild the EXACT same loop as a LangGraph
graph — wiring it by hand with StateGraph so you can see every node and edge:

        ┌──────────────────────────────────────────┐
        │                                            │
   START ──► llm_node ──(wants a tool?)──► tool_node ┘   (loop back to llm)
                  │
                  └──(no tool, just an answer)──► END

The model THINKS in llm_node, ACTS in tool_node, then loops back to think again
with the tool results in hand. That cycle — impossible in a linear LCEL chain —
is the whole reason agents need a graph.

Run:  python agent.py
"""

import os
from dotenv import load_dotenv

# ── IMPORTS ───────────────────────────────────────────────────────────────────
from typing import Annotated, TypedDict
# StateGraph = the builder; START/END = entry/exit sentinels (core_concepts §5,§8)
from langgraph.graph import StateGraph, START, END
# add_messages = the reducer that APPENDS new messages instead of overwriting (§2)
from langgraph.graph.message import add_messages
# InMemorySaver = the checkpointer that gives native memory per thread_id (§7)
from langgraph.checkpoint.memory import InMemorySaver
# Tools + message types — same as tool_notes.py
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq

# Load GROQ_API_KEY from phase1/.env (path relative to THIS file).
HERE = os.path.dirname(os.path.abspath(__file__))         # .../phase2/langgraph
ROOT = os.path.dirname(os.path.dirname(HERE))             # .../agentic-ai-journey
load_dotenv(dotenv_path=os.path.join(ROOT, "phase1", ".env"))


# ── 1. TOOLS ──────────────────────────────────────────────────────────────────
# Plain functions exposed to the LLM. The docstring + type hints become the spec
# the model reads to decide WHEN and HOW to call each (see tool_notes.py §1).
# We give it three so the agent has a real choice to make.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together and return the product."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two integers together and return the sum."""
    return a + b

@tool
def word_length(word: str) -> int:
    """Return the number of characters in a single word."""
    return len(word)

tools = [multiply, add, word_length]
tools_by_name = {t.name: t for t in tools}      # name → function, for the tool node


# ── 2. THE MODEL (tool-aware) ─────────────────────────────────────────────────
# bind_tools attaches the tool specs so the model can REQUEST them. temperature=0
# for deterministic tool choices. This same llm is used inside llm_node below.
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).bind_tools(tools)


# ── 3. STATE ──────────────────────────────────────────────────────────────────
# The shared data flowing through the graph. We only need the running message
# list. The add_messages reducer means every node that returns {"messages": [...]}
# APPENDS to the conversation rather than replacing it — so history accumulates
# automatically as we loop. (core_concepts.md §2-3)
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ── 4. NODES (plain functions: whole state in, partial state out) ─────────────
# llm_node = the "THINK" step. Feed the full conversation to the model; it returns
# an AIMessage that either contains a final answer OR carries .tool_calls (a tool
# request). We append that message to state.
def llm_node(state: State) -> dict:
    response = llm.invoke(state["messages"])        # one model round-trip
    return {"messages": [response]}                 # append it (reducer handles merge)

# tool_node = the "ACT" step. The most recent message is the model's tool request.
# We execute every requested tool ourselves (the model never runs code — §3 of
# tool_notes) and append one ToolMessage per call, linked by tool_call_id.
def tool_node(state: State) -> dict:
    last_message = state["messages"][-1]            # the AIMessage with tool_calls
    results = []
    for call in last_message.tool_calls:            # there may be several
        chosen = tools_by_name[call["name"]]        # look up the real function
        output = chosen.invoke(call["args"])        # WE run it
        print(f"   [tool] {call['name']}({call['args']}) = {output}")
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}                    # append results to history


# ── 5. THE ROUTER (conditional edge logic) ────────────────────────────────────
# After llm_node runs, look at its last message. If the model asked for a tool,
# go to tool_node. If not, it produced a final answer → END. This single function
# is what creates the LOOP vs EXIT branch. (core_concepts.md §4)
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:                     # model wants a tool
        return "tools"
    return END                                      # model is done


# ── 6. BUILD THE GRAPH ────────────────────────────────────────────────────────
builder = StateGraph(State)                         # graph built around our State
builder.add_node("llm", llm_node)                   # register the THINK node
builder.add_node("tools", tool_node)                # register the ACT node

builder.add_edge(START, "llm")                      # entry: always start by thinking
# Conditional edge OUT of llm: router decides "tools" (loop) or END (exit).
# The dict maps each possible router return value → the destination node.
builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "llm")                    # THE LOOP: after acting, think again


# ── 7. COMPILE WITH MEMORY ────────────────────────────────────────────────────
# The checkpointer saves state after every step, keyed by thread_id. This is the
# NATIVE replacement for RunnableWithMessageHistory + InMemoryChatMessageHistory
# you used in conversational_rag.py — no manual history plumbing at all. (§7)
agent = builder.compile(checkpointer=InMemorySaver())


# ── 8. RUN IT ─────────────────────────────────────────────────────────────────
def chat(question, thread_id="demo"):
    # Same thread_id = same remembered conversation. invoke runs START→…→END,
    # looping llm↔tools as many times as the model needs.
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [HumanMessage(question)]}, config)
    print(f"Q: {question}")
    print(f"A: {result['messages'][-1].content}\n" + "-" * 78)

if __name__ == "__main__":
    # (a) Single tool call: llm → tools → llm → END
    chat("What is 23 multiplied by 17?")

    # (b) MULTI-STEP, needs the loop to run twice (add, then multiply the result):
    #     llm → tools(add) → llm → tools(multiply) → llm → END
    chat("Add 8 and 5, then multiply the result by 10.")

    # (c) MEMORY: a follow-up with no tool keywords — only works because the
    #     checkpointer kept turn (b) on this same thread_id.
    chat("What was the final number you just gave me, doubled?")

    # (d) No tool needed at all: llm → END (router exits immediately)
    chat("Say hi in one short sentence.")
