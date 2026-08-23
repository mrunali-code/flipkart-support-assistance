"""
Flipkart Support Agent Prompt Engineering (Task 6) and MOCK_LLM Generator (Task 7).

Applies 4S Principles:
- Specific: Explicit domain (Flipkart customer support, returns, SLAs, risk scoring, image classification), strict JSON output schema.
- Short: Compact instructions without conversational filler.
- Surround: Clear XML tags (<SYSTEM_ROLE>, <INTENT_RULES>, <OUTPUT_FORMAT>, <CONTEXT>) framing the prompt.
- Single: One clear goal per node (classify intent, generate grounded JSON answer).

Includes few-shot examples for intent classification and deterministic MOCK_LLM rule-based generator.
"""

import json
from typing import Dict, Any, List

SYSTEM_PROMPT_INTENT = """<SYSTEM_ROLE>
You are the intent classification module for Flipkart Customer Support.
Classify the user input into exactly one of three intents:
1. "policy_rag" - Questions about Flipkart return policies, delivery SLAs, COD refunds, reverse pickups, or standard customer service rules.
2. "return_risk" - Questions asking to evaluate, check, or predict order return risk / return probability for specific order IDs or order features.
3. "image_classify" - Questions asking to identify, categorize, or inspect product images from data/sample_images/.
</SYSTEM_ROLE>

<INTENT_RULES>
Analyze user query and short-term memory state.
Output only a valid JSON with key "intent".
</INTENT_RULES>

<FEW_SHOT_EXAMPLES>
Example 1:
User: "How many days do I have to return running shoes?"
Output: {"intent": "policy_rag"}

Example 2:
User: "Can you check if order 104 is likely to be returned?"
Output: {"intent": "return_risk"}

Example 3:
User: "What category is this product image data/sample_images/00_ankle_boot.png?"
Output: {"intent": "image_classify"}

Example 4:
User: "How long does a COD refund take to hit my bank account?"
Output: {"intent": "policy_rag"}
</FEW_SHOT_EXAMPLES>
"""

SYSTEM_PROMPT_RESPONSE_GENERATOR = """<SYSTEM_ROLE>
You are Flipkart's AI Support Assistant. You provide accurate, friendly, and strictly grounded answers to customer queries.
</SYSTEM_ROLE>

<GUIDELINES>
- Specific: Only answer based on retrieved policy context or tool results.
- Short: Keep answers concise and direct (2-4 sentences max).
- Surround: Context is provided within <RETRIEVED_CONTEXT> or <TOOL_OUTPUT>.
- Single: Output must strictly conform to the JSON schema below.
</GUIDELINES>

<OUTPUT_FORMAT>
Return ONLY a valid JSON object matching this schema:
{
  "answer": "String answering the user's question clearly and politely.",
  "source": "policy_kb" | "return_risk_tool" | "image_classifier_tool",
  "confidence": Float (between 0.0 and 1.0)
}
</OUTPUT_FORMAT>
"""

def mock_llm_intent_classifier(user_query: str, state: Dict[str, Any]) -> str:
    """
    Deterministic rule-based intent classifier.
    Returns: 'policy_rag', 'return_risk', or 'image_classify'.
    """
    q_lower = user_query.lower()
    
    # Check for image classification cues
    if any(k in q_lower for k in [".png", ".jpg", "image", "classify product", "category of this item", "sample_images", "what product is this"]):
        return "image_classify"
        
    # Check for return risk cues
    if any(k in q_lower for k in ["return risk", "risk score", "return probability", "risk bucket", "likely to return", "check order", "order risk", "order_id"]):
        return "return_risk"
        
    # Check if user query refers to previously discussed order in conversational state
    if state.get("last_order_id") and any(k in q_lower for k in ["that order", "this order", "its risk", "will it be returned", "what about the risk"]):
        return "return_risk"
        
    # Default to policy RAG
    return "policy_rag"

def mock_llm_response_generator(intent: str, query: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task 7: Deterministic rule-based template generator for final structured response.
    Zero network calls, zero external API keys.
    Conforms strictly to fixed JSON schema:
    {
        "answer": str,
        "source": "policy_kb" | "return_risk_tool" | "image_classifier_tool",
        "confidence": float
    }
    """
    if intent == "policy_rag":
        retrieved_chunks = context_data.get("retrieved_chunks", [])
        if not retrieved_chunks:
            return {
                "answer": "I apologize, but I could not find relevant policy information to answer your query. Please contact Flipkart customer care for further assistance.",
                "source": "policy_kb",
                "confidence": 0.0
            }
        
        top_chunk = retrieved_chunks[0]
        similarity = top_chunk.get("similarity_score", 0.85)
        doc_title = top_chunk.get("doc_title", "Flipkart Policy")
        chunk_text = top_chunk.get("text", "")
        full_content = top_chunk.get("full_doc_content", chunk_text)
        
        answer = f"According to Flipkart's {doc_title}: {full_content}"
        return {
            "answer": answer,
            "source": "policy_kb",
            "confidence": round(float(similarity), 4)
        }
        
    elif intent == "return_risk":
        risk_result = context_data.get("tool_result", {})
        order_id = risk_result.get("order_id", "N/A")
        prob = risk_result.get("return_probability", 0.0)
        bucket = risk_result.get("risk_bucket", "Low")
        t_star = risk_result.get("t_star_rf", 0.50)
        
        answer = (
            f"Order #{order_id} has a predicted return probability of {prob:.2%} "
            f"and is classified into the '{bucket}' risk bucket (calibrated against model threshold t* = {t_star:.2f})."
        )
        return {
            "answer": answer,
            "source": "return_risk_tool",
            "confidence": round(float(prob if bucket != 'Low' else 1.0 - prob), 4)
        }
        
    elif intent == "image_classify":
        img_result = context_data.get("tool_result", {})
        pred_class = img_result.get("predicted_class", "Unknown")
        conf = img_result.get("confidence", 0.90)
        
        answer = f"The uploaded product image is classified as '{pred_class}' with {conf:.2%} model confidence."
        return {
            "answer": answer,
            "source": "image_classifier_tool",
            "confidence": round(float(conf), 4)
        }
        
    else:
        return {
            "answer": "How can I assist you with Flipkart orders, policies, or product questions today?",
            "source": "policy_kb",
            "confidence": 1.0
        }
