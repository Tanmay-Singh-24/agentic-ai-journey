"""
═══════════════════════════════════════════════════════════════════════════════
 TOOL NOTES — giving an LLM the ability to call Python functions (LangChain 1.x)
═══════════════════════════════════════════════════════════════════════════════

WHY TOOLS EXIST
An LLM only produces text. It cannot do math reliably, look up today's date,
hit an API, or read a database. A "tool" is just a normal Python function that
we expose to the LLM so it can ask for it to be run.

THE ONE IDEA TO BURN IN:
  The LLM NEVER runs your code. It only REQUESTS a call — it outputs JSON saying
  "please run multiply with a=23, b=17". YOUR code executes the function and
  hands the result back. The model decides WHAT to call; you do the calling.

THE TOOL-CALLING LOOP (the whole mechanism):
  1. You bind tools to the LLM (tell it what functions exist).
  2. You send a question.
  3. The LLM replies either with a normal answer OR with tool_calls (requests).
  4. You execute each requested tool and send the results back as ToolMessages.
  5. The LLM reads the results and writes the final answer.
  (Step 3→5 can repeat many times. An "agent" = this loop run automatically.)

Run:  python tool_notes.py
"""

import os
from dotenv import load_dotenv

# ── IMPORTS ───────────────────────────────────────────────────────────────────
# @tool       — decorator that turns a plain function into a LangChain Tool.
from langchain_core.tools import tool
# Message types we pass to / get back from a chat model:
#   HumanMessage  — a user turn.
#   AIMessage     — the model's reply (may carry .tool_calls instead of text).
#   ToolMessage   — the RESULT of running a tool, sent back to the model. It must
#                   reference the tool_call id the model gave us, so the model
#                   knows which request this answers.
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq

# Load GROQ_API_KEY from phase1/.env (path relative to THIS file so it runs anywhere).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
load_dotenv(dotenv_path=os.path.join(ROOT, "phase1", ".env"))


# ── 1. DEFINING A TOOL ────────────────────────────────────────────────────────
# The @tool decorator reads THREE things and turns them into a spec the LLM sees:
#   • the function NAME            → becomes the tool's name
#   • the DOCSTRING               → becomes the tool's description (how the LLM
#                                    decides WHEN to use it — write it carefully!)
#   • the TYPE HINTS (a: int ...) → become the argument schema (what the LLM must
#                                    fill in, and what types are expected)
# So the docstring + type hints ARE the interface the model reads. Vague docstring
# = the model picks the wrong tool or wrong args.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together and return the product."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two integers together and return the sum."""
    return a + b


@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a single word."""
    return len(word)


# A tool is still callable, plus it now carries metadata. Inspect what the LLM sees:
print("── 1. WHAT A TOOL LOOKS LIKE ─────────────────────────────")
print("name        :", multiply.name)          # "multiply"
print("description :", multiply.description)    # the docstring
print("args        :", multiply.args)          # {'a': {...int}, 'b': {...int}}
# You invoke a tool with a dict of args (NOT positional). This is how WE run it
# after the model requests it.
print("manual call :", multiply.invoke({"a": 6, "b": 7}))   # -> 42
print()


# ── 2. BINDING TOOLS TO THE LLM ───────────────────────────────────────────────
# bind_tools() does NOT change the model — it returns a NEW model object that
# sends the tool specs along with every request, so the model knows these
# functions exist and may request them. temperature=0 → deterministic choices.
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
tools = [multiply, add, get_word_length]
llm_with_tools = llm.bind_tools(tools)         # the tool-aware model

# A lookup table so we can find the real function by the name the model returns.
tools_by_name = {t.name: t for t in tools}


# ── 3. ASKING A QUESTION → SEEING THE REQUEST (not the answer yet) ────────────
# The model does NOT answer "23 * 17" itself. It returns an AIMessage whose
# .content is usually empty and whose .tool_calls lists what it wants run.
print("── 3. THE MODEL REQUESTS A TOOL ──────────────────────────")
question = "What is 23 multiplied by 17?"
messages = [HumanMessage(question)]            # conversation so far (a list!)
ai_msg = llm_with_tools.invoke(messages)       # first round-trip to the model

# .tool_calls is a list of dicts: {"name", "args", "id", "type"}.
#   name → which tool, args → arguments to pass, id → ties the result back later.
print("content   :", repr(ai_msg.content))     # often '' when a tool is requested
print("tool_calls:", ai_msg.tool_calls)
print()


# ── 4. EXECUTING THE TOOL AND REPLYING WITH THE RESULT ────────────────────────
# We MUST append the model's ai_msg to history first (it records the request),
# then add one ToolMessage per executed call, matched by tool_call_id.
print("── 4. WE RUN THE TOOL, THEN THE MODEL ANSWERS ────────────")
messages.append(ai_msg)                         # record the model's request turn

for call in ai_msg.tool_calls:                  # there could be several
    chosen_tool = tools_by_name[call["name"]]   # look up the real function
    result = chosen_tool.invoke(call["args"])   # WE execute it here
    print(f"ran {call['name']}{call['args']} = {result}")
    # Send the result back, linked to the exact request via its id:
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

# Second round-trip: now the model has the tool result and writes a real answer.
final = llm_with_tools.invoke(messages)
print("final answer:", final.content)
print()


# ── 5. THE AUTOMATIC LOOP (this is the seed of an "agent") ────────────────────
# Doing step 3-4 by hand gets tedious, and some questions need MULTIPLE tool calls
# in sequence (e.g. "add 5 and 3, then multiply the result by 10"). So we wrap the
# request→execute→reply cycle in a while-loop that keeps going until the model
# stops asking for tools and returns plain text. That loop IS what a LangChain
# agent automates for you — you'll meet the official version next.
def run_with_tools(user_question, max_steps=5):
    msgs = [HumanMessage(user_question)]        # start the conversation
    for _ in range(max_steps):                  # cap steps so we can't loop forever
        ai = llm_with_tools.invoke(msgs)        # ask the model what to do
        msgs.append(ai)                         # record its turn
        if not ai.tool_calls:                   # no tool requested → it's the answer
            return ai.content
        for call in ai.tool_calls:              # otherwise run every requested tool
            out = tools_by_name[call["name"]].invoke(call["args"])
            msgs.append(ToolMessage(content=str(out), tool_call_id=call["id"]))
    return "Stopped: hit max_steps without a final answer."

print("── 5. AUTOMATIC MULTI-STEP LOOP ──────────────────────────")
# This needs TWO tools in order: add(5,3)=8, then multiply(8,10)=80.
print(run_with_tools("Add 5 and 3, then multiply that result by 10."))
# This one needs no tool at all — the model just answers directly.
print(run_with_tools("Say hello in one short sentence."))
print()


# ── 6. QUICK REFERENCE / GOTCHAS ──────────────────────────────────────────────
# • The docstring is the model's instruction manual — be precise and specific.
# • Type hints define the arg schema; without them the model can't fill args right.
# • llm.bind_tools(tools) returns a NEW model; the original llm is unchanged.
# • The model REQUESTS (ai_msg.tool_calls); YOUR code EXECUTES (tool.invoke(args)).
# • Every ToolMessage must carry the matching tool_call_id, or the model gets lost.
# • Always append the model's tool-request AIMessage to history BEFORE the
#   ToolMessages — the request and its result must sit together, in order.
# • Cap the loop (max_steps) so a confused model can't call tools forever.
print("── 6. See the comments above for the cheat-sheet. ────────")
