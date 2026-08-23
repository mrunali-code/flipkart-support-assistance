"""
Flipkart Support Agent using LangGraph (Part 3 Task 5).

Constructs a 4-node state graph:
1. intent_node (decides policy_rag vs return_risk vs image_classify, incorporates input guardrail)
2. rag_retrieval_node (retrieves relevant policy chunks and applies output groundedness check)
3. tool_calling_node (executes check_return_risk or classify_product_image with real artifacts)
4. response_generation_node (deterministic MOCK_LLM structured JSON generation)

Conditional branching routes queries based on classified intent.
Maintains short-term conversational state across multi-turn exchanges and resets cleanly on fresh invocations.
"""

import os
import re
import json
import pandas as pd
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from part3_vector_store import PolicyVectorStore
from part3_tools import check_return_risk, classify_product_image
from part3_guardrails import check_input_guardrail, check_output_groundedness, GROUNDEDNESS_SIMILARITY_THRESHOLD
from part3_prompts_mock_llm import mock_llm_intent_classifier, mock_llm_response_generator

# Global vector store instance
_vector_store = None
def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = PolicyVectorStore()
    return _vector_store

# Orders DataFrame cache for order lookup by ID
_orders_df = None
def get_orders_df():
    global _orders_df
    if _orders_df is None:
        if os.path.exists("orders_dataset.csv"):
            _orders_df = pd.read_csv("orders_dataset.csv")
    return _orders_df

# Define Agent State
class AgentState(TypedDict):
    query: str
    conversation_history: List[Dict[str, str]]
    last_order_id: Optional[int]
    last_image_path: Optional[str]
    intent: Optional[str]
    guardrail_blocked: bool
    guardrail_message: Optional[str]
    groundedness_refusal: bool
    groundedness_details: Optional[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    final_response: Optional[Dict[str, Any]]

# ============================================================
# Node 1: Intent Node
# ============================================================
def intent_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "")
    
    # 1. Input-side Prompt Injection Guardrail Check
    is_safe, flag_reason = check_input_guardrail(query)
    if not is_safe:
        return {
            "intent": "blocked",
            "guardrail_blocked": True,
            "guardrail_message": f"Security Alert: Your request was blocked by input guardrails ({flag_reason})."
        }
    
    # 2. Extract potential Order ID or Image Path from query to update conversational state
    updates: Dict[str, Any] = {
        "guardrail_blocked": False,
        "guardrail_message": None
    }
    
    # Check for order id pattern
    order_id_match = re.search(r"order\s*(?:id|#)?\s*[:=]?\s*(\d+)", query, re.IGNORECASE)
    if order_id_match:
        updates["last_order_id"] = int(order_id_match.group(1))
    
    # Check for image path pattern
    img_match = re.search(r"(data/sample_images/[\w\-]+\.png|sample_images/[\w\-]+\.png|[\w\-]+\.png)", query, re.IGNORECASE)
    if img_match:
        updates["last_image_path"] = img_match.group(1)
        
    # Classify intent using prompt/mock classifier
    current_state = dict(state)
    current_state.update(updates)
    intent = mock_llm_intent_classifier(query, current_state)
    updates["intent"] = intent
    
    return updates

# ============================================================
# Node 2: RAG Retrieval Node
# ============================================================
def rag_retrieval_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "")
    vector_store = get_vector_store()
    
    # Retrieve top 3 relevant chunks
    chunks = vector_store.search(query, top_k=3)
    
    # Output-side Groundedness Check
    is_grounded, max_sim, reason = check_output_groundedness(chunks, threshold=GROUNDEDNESS_SIMILARITY_THRESHOLD)
    
    if not is_grounded:
        return {
            "retrieved_chunks": chunks,
            "groundedness_refusal": True,
            "groundedness_details": {
                "max_similarity": max_sim,
                "threshold": GROUNDEDNESS_SIMILARITY_THRESHOLD,
                "reason": reason
            }
        }
    else:
        return {
            "retrieved_chunks": chunks,
            "groundedness_refusal": False,
            "groundedness_details": {
                "max_similarity": max_sim,
                "threshold": GROUNDEDNESS_SIMILARITY_THRESHOLD,
                "reason": reason
            }
        }

# ============================================================
# Node 3: Tool Calling Node
# ============================================================
def tool_calling_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("intent")
    query = state.get("query", "")
    
    if intent == "return_risk":
        # Extract order id from current query or prior turn state
        order_id = state.get("last_order_id")
        orders_df = get_orders_df()
        
        # Default / fallback order features if order ID not in CSV
        order_features = {
            "order_id": order_id if order_id else 4,
            "product_category": "Home",
            "price_inr": 7930.0,
            "discount_pct": 49.7,
            "payment_method": "COD",
            "customer_tenure_days": 1479,
            "num_previous_orders": 33,
            "num_previous_returns": 4,
            "delivery_distance_km": 143.1,
            "delivery_days": 5,
            "is_weekend_order": 0,
            "rating_given": 3.0
        }
        
        if orders_df is not None and order_id is not None:
            match = orders_df[orders_df["order_id"] == order_id]
            if len(match) > 0:
                row = match.iloc[0]
                order_features = {
                    "order_id": order_id,
                    "product_category": row["product_category"],
                    "price_inr": float(row["price_inr"]),
                    "discount_pct": float(row["discount_pct"]),
                    "payment_method": row["payment_method"],
                    "customer_tenure_days": int(row["customer_tenure_days"]),
                    "num_previous_orders": int(row["num_previous_orders"]),
                    "num_previous_returns": int(row["num_previous_returns"]),
                    "delivery_distance_km": float(row["delivery_distance_km"]),
                    "delivery_days": int(row["delivery_days"]),
                    "is_weekend_order": int(row["is_weekend_order"]),
                    "rating_given": float(row["rating_given"])
                }
        
        tool_result = check_return_risk(order_features)
        return {"tool_result": tool_result}
        
    elif intent == "image_classify":
        img_path = state.get("last_image_path")
        if not img_path or not os.path.exists(img_path):
            # Check standard path
            if img_path and os.path.exists(os.path.join("data/sample_images", os.path.basename(img_path))):
                img_path = os.path.join("data/sample_images", os.path.basename(img_path))
            else:
                img_path = "data/sample_images/00_ankle_boot.png"
                
        tool_result = classify_product_image(img_path)
        return {"tool_result": tool_result, "last_image_path": img_path}
        
    return {"tool_result": None}

# ============================================================
# Node 4: Response Generation Node
# ============================================================
def response_generation_node(state: AgentState) -> Dict[str, Any]:
    # Check if blocked by input guardrail
    if state.get("guardrail_blocked"):
        return {
            "final_response": {
                "answer": state.get("guardrail_message", "Request blocked due to security policy violation."),
                "source": "policy_kb",
                "confidence": 0.0
            }
        }
        
    # Check if refused by output groundedness check
    if state.get("groundedness_refusal"):
        details = state.get("groundedness_details", {})
        max_sim = details.get("max_similarity", 0.0)
        thresh = details.get("threshold", GROUNDEDNESS_SIMILARITY_THRESHOLD)
        return {
            "final_response": {
                "answer": (
                    f"Refusal: I cannot answer this policy question because no sufficiently similar policy document was found "
                    f"(Retrieved similarity score: {max_sim:.4f}, Required groundedness threshold: {thresh:.4f}). "
                    f"Please consult Flipkart Customer Support directly."
                ),
                "source": "policy_kb",
                "confidence": round(max_sim, 4)
            }
        }
        
    intent = state.get("intent", "policy_rag")
    query = state.get("query", "")
    context_data = {
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "tool_result": state.get("tool_result", {})
    }
    
    resp = mock_llm_response_generator(intent, query, context_data)
    
    # Update conversation history
    history = list(state.get("conversation_history", []))
    history.append({"user": query, "agent": resp["answer"]})
    
    return {
        "final_response": resp,
        "conversation_history": history
    }

# ============================================================
# Conditional Edge Router
# ============================================================
def route_intent(state: AgentState) -> str:
    if state.get("guardrail_blocked"):
        return "response_gen"
    intent = state.get("intent")
    if intent == "policy_rag":
        return "rag"
    elif intent in ["return_risk", "image_classify"]:
        return "tools"
    else:
        return "response_gen"

# ============================================================
# Build LangGraph Agent Graph
# ============================================================
def build_support_agent_graph():
    builder = StateGraph(AgentState)
    
    # Add 4 Nodes
    builder.add_node("intent", intent_node)
    builder.add_node("rag", rag_retrieval_node)
    builder.add_node("tools", tool_calling_node)
    builder.add_node("response_gen", response_generation_node)
    
    # Set Entry Point
    builder.set_entry_point("intent")
    
    # Conditional Edges from Intent Node
    builder.add_conditional_edges(
        "intent",
        route_intent,
        {
            "rag": "rag",
            "tools": "tools",
            "response_gen": "response_gen"
        }
    )
    
    # Edges leading to Response Generation Node
    builder.add_edge("rag", "response_gen")
    builder.add_edge("tools", "response_gen")
    builder.add_edge("response_gen", END)
    
    return builder.compile()

# Interactive Session Helper for Multi-turn Conversations
class SupportAgentSession:
    def __init__(self):
        self.graph = build_support_agent_graph()
        self.state: AgentState = {
            "query": "",
            "conversation_history": [],
            "last_order_id": None,
            "last_image_path": None,
            "intent": None,
            "guardrail_blocked": False,
            "guardrail_message": None,
            "groundedness_refusal": False,
            "groundedness_details": None,
            "retrieved_chunks": [],
            "tool_result": None,
            "final_response": None
        }
        
    def reset(self):
        """Resets conversational state for fresh conversations."""
        self.state = {
            "query": "",
            "conversation_history": [],
            "last_order_id": None,
            "last_image_path": None,
            "intent": None,
            "guardrail_blocked": False,
            "guardrail_message": None,
            "groundedness_refusal": False,
            "groundedness_details": None,
            "retrieved_chunks": [],
            "tool_result": None,
            "final_response": None
        }
        
    def chat(self, user_query: str) -> Dict[str, Any]:
        """Runs one turn of conversation through the graph."""
        self.state["query"] = user_query
        # Reset per-turn transient state while keeping long-term memory (conversation_history, last_order_id, last_image_path)
        self.state["guardrail_blocked"] = False
        self.state["guardrail_message"] = None
        self.state["groundedness_refusal"] = False
        self.state["groundedness_details"] = None
        self.state["retrieved_chunks"] = []
        self.state["tool_result"] = None
        self.state["final_response"] = None
        self.state["intent"] = None
        
        self.state = self.graph.invoke(self.state)
        return self.state["final_response"]

if __name__ == "__main__":
    session = SupportAgentSession()
    print("Testing Support Agent Graph...")
    res = session.chat("What is the return window for footwear?")
    print("Turn 1 Response:", json.dumps(res, indent=2))
