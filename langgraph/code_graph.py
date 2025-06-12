# flake8: noqa

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

client = OpenAI()


class ClassifyMessageResponse(BaseModel):
    is_coding_question: bool


class CodeAccuracyResponse(BaseModel):
    accuracy_percentage: str


class State(TypedDict):
    user_query: str
    llm_result: str | None
    accuracy_percentage: str | None
    is_coding_question: bool | None


def classify_message(state: State):
    print("⚠️ classify_message")
    query = state["user_query"]
    
    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is
    related to coding question or not.
    Return the response in specified JSON boolean only.
    """

    # Structured Outputs / Responses
    response = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        response_format=ClassifyMessageResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )

    is_coding_question = response.choices[0].message.parsed.is_coding_question
    state["is_coding_question"] = is_coding_question

    return state

# routing karte waqt ek cheez hai aapko yaha se ek cheez return karni hoti hai that is literal (->) kyunke usko jaanna hai ke aap isko kaha kaha route kar sakte ho 


def route_query(state: State) -> Literal["general_query", "coding_query"]:
    print("⚠️ route_query")
    is_coding = state["is_coding_question"]

    if is_coding:
        return "coding_query"

    return "general_query"


def general_query(state: State):
    print("⚠️ general_query")
    query = state["user_query"]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": query}
        ]
    )

    state["llm_result"] = response.choices[0].message.content

    return state


def coding_query(state: State):
    print("⚠️ coding_query")
    query = state["user_query"]

    SYSTEM_PROMPT = """
        You are a Coding Expert Agent
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )

    state["llm_result"] = response.choices[0].message.content

    return state


def coding_validate_query(state: State):
    print("⚠️ coding_validate_query")
    query = state["user_query"]
    llm_code = state["llm_result"]

    SYSTEM_PROMPT = f"""
        You are expect in calculating accuracy of the code according to the question.
        Return the percentage of accuracy
        
        User Query: {query},
        Code: {llm_code}
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        response_format=CodeAccuracyResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )

    state["accuracy_percentage"] = response.choices[0].message.parsed.accuracy_percentage
    return state


def conditional_edge(state: State) -> Literal["coding_query", "__end__"]:
    print("⚠️ conditional_edge")

    raw = state["accuracy_percentage"]
    if raw is None:
        return "coding_query"

    numeric = int(raw.strip('%'))

    if numeric < 95:
        print(f"🔁 Accuracy {numeric}% is below threshold, retrying...")
        return "coding_query"
    else:
        print(f"✅ Accuracy {numeric}% is sufficient. Ending.")
        return "__end__"


graph_builder = StateGraph(State)

# Define Nodes
graph_builder.add_node("classify_message", classify_message)
graph_builder.add_node("route_query", route_query)
graph_builder.add_node("general_query", general_query)
graph_builder.add_node("coding_query", coding_query)
graph_builder.add_node("coding_validate_query", coding_validate_query)

graph_builder.add_edge(START, "classify_message")
graph_builder.add_conditional_edges("classify_message", route_query)

graph_builder.add_edge("route_query", END)
graph_builder.add_edge("coding_query", "coding_validate_query")

graph_builder.add_conditional_edges("coding_validate_query", conditional_edge)


graph = graph_builder.compile()


def main():
    user = input("> ")
    
    _state: State = {
        "user_query": user,
        "accuracy_percentage": None,
        "is_coding_question": False,
        "llm_result": None
    }
    
    response = graph.invoke(_state)
    
    print("🧾 Final Output:", response)
    

def test_conditional_edge_loop():
    state = {
        "user_query": "Write a function to reverse a string in Python",
        "llm_result": "def reverse(s): return s[::-1]",
        "accuracy_percentage": "90%",
        "is_coding_question": True
    }

    max_attempts = 5
    attempt = 1

    while attempt <= max_attempts:
        print(
            f"\n🧪 Attempt {attempt}: Accuracy = {state['accuracy_percentage']}")
        next_node = conditional_edge(state)

        if next_node == END:
            print("✅ Done.")
            break
        else:
            attempt += 1
            improved = 90 + attempt
            state["accuracy_percentage"] = f"{improved}%"
            state["llm_result"] = f"# Improved code attempt {attempt}"

    else:
        print("❌ Max attempts reached without sufficient accuracy.")


if __name__ == "__main__":
    main()
