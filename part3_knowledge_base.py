"""
Policy Knowledge Base and Retrieval Evaluation Key for Flipkart Support Agent.

Contains 14 short (2-4 sentence) Flipkart-style policy documents across categories:
- Return windows by category (Apparel, Footwear, Electronics, Home, Grocery, Beauty)
- COD Refund Timelines and Modes
- Delivery SLAs (Standard, Express, Metro, Rural)
- Reverse-Pickup Eligibility and Quality Check Conditions
"""

import json
import re
from typing import List, Dict, Tuple

# 14 Short Flipkart-style policy documents
POLICY_DOCUMENTS = [
    {
        "doc_id": "DOC_01_APPAREL_RETURN_WINDOW",
        "title": "Return Window for Lifestyle and Apparel",
        "category": "return_windows",
        "content": "Flipkart offers a 14-day hassle-free return window for all lifestyle and apparel products starting from the date of delivery. Items must be unused, unwashed, and retained with original tags and packaging. Customers can request either an exchange for size/color or a full refund to the original payment source."
    },
    {
        "doc_id": "DOC_02_FOOTWEAR_RETURN_WINDOW",
        "title": "Return Window and Policy for Footwear",
        "category": "return_windows",
        "content": "Footwear products including shoes, sandals, and boots are eligible for return within 10 days of delivery. The footwear must show no signs of outdoor wear, and the original brand shoe box must be intact. Both replacement of size and monetary refunds are supported during reverse pickup."
    },
    {
        "doc_id": "DOC_03_ELECTRONICS_RETURN_WINDOW",
        "title": "Return and Replacement Window for Electronics",
        "category": "return_windows",
        "content": "Electronics such as smartphones, laptops, and tablets carry a strict 7-day replacement-only window for defective or physically damaged items upon arrival. Direct refunds are not issued for electronics unless an authorized technician verifies the defect on-site and replacement inventory is out of stock. Serial numbers and IMEI numbers must match the invoice."
    },
    {
        "doc_id": "DOC_04_HOME_APPLIANCES_RETURN_WINDOW",
        "title": "Return Policy for Home Furnishing and Appliances",
        "category": "return_windows",
        "content": "Home furnishing and small kitchen appliances have a 10-day return window from delivery. Large appliances such as refrigerators and washing machines qualify for a 10-day technician doorstep visit for repair or replacement rather than direct return. Original manuals, components, and accessories must be returned together."
    },
    {
        "doc_id": "DOC_05_GROCERY_NON_RETURNABLE",
        "title": "Policy for Grocery, Food, and Personal Care Items",
        "category": "return_windows",
        "content": "Grocery and perishable food items are strictly non-returnable once accepted at delivery due to hygiene standards. In cases of damaged or expired food delivery, customers must report within 24 hours to claim an instant refund or coupon credit. Personal hygiene products and cosmetics are similarly non-returnable once the factory seal is broken."
    },
    {
        "doc_id": "DOC_06_COD_REFUND_TIMELINE_BANK",
        "title": "COD Refund Timelines via Direct Bank Transfer (NEFT/IMPS)",
        "category": "cod_refunds",
        "content": "For orders paid via Cash on Delivery (COD), refunds are processed via NEFT bank transfer within 2 to 5 business days after the returned product clears quality check. Customers must add and verify their bank account or UPI ID in the Flipkart refund settings page. Notifications with UTR transaction details are sent via SMS and email."
    },
    {
        "doc_id": "DOC_07_COD_REFUND_INSTANT_WALLET",
        "title": "Instant COD Refund via Flipkart Gift Card or Wallet",
        "category": "cod_refunds",
        "content": "Customers who choose Flipkart Gift Card or SuperCoins wallet as their refund mode for COD orders receive their refund instantly within 2 hours of reverse pickup completion. Gift card balances have a 1-year validity and can be used on any future Flipkart transaction without restrictions. This mode avoids banking settlement delays."
    },
    {
        "doc_id": "DOC_08_DELIVERY_SLA_METRO",
        "title": "Delivery SLAs for Metro and Tier-1 Cities",
        "category": "delivery_slas",
        "content": "Standard delivery service level agreements (SLAs) for metro and Tier-1 cities range from 1 to 3 business days from order placement. Flipkart Plus members in select pin codes enjoy free Next-Day or Same-Day delivery on eligible F-Assured items. Orders placed before 12 PM noon qualify for same-day dispatch."
    },
    {
        "doc_id": "DOC_09_DELIVERY_SLA_TIER2_RURAL",
        "title": "Delivery SLAs for Tier-2, Tier-3, and Remote Pin Codes",
        "category": "delivery_slas",
        "content": "Deliveries to Tier-2, Tier-3 towns and rural/remote pin codes take between 4 to 8 business days depending on road transit connectivity. Regional warehouse logistics partners ensure tracking updates at every sorting hub along the route. During national sale events or adverse weather conditions, delivery timelines may extend by 2 additional days."
    },
    {
        "doc_id": "DOC_10_REVERSE_PICKUP_DOORSTEP_QC",
        "title": "Reverse-Pickup Doorstep Quality Check (QC) Criteria",
        "category": "reverse_pickup",
        "content": "During reverse-pickup, the delivery executive performs a mandatory visual Doorstep Quality Check (QC) before accepting the returned parcel. For apparel and footwear, the executive verifies that tags are attached, fabric is unstained, and brand packaging is present. If the item fails QC criteria, reverse pickup is immediately cancelled."
    },
    {
        "doc_id": "DOC_11_REVERSE_PICKUP_ATTEMPTS_SLA",
        "title": "Reverse-Pickup Scheduling, Attempts, and Timelines",
        "category": "reverse_pickup",
        "content": "Reverse pickups are scheduled within 24 to 48 hours of return approval for serviceable addresses. The courier partner makes up to three pickup attempts before the return request is cancelled automatically. Customers receive an SMS containing an OTP that must be shared with the courier agent to authorize parcel collection."
    },
    {
        "doc_id": "DOC_12_PREPAID_REFUND_TIMELINES",
        "title": "Refund Timelines for Prepaid Transactions (Credit Card, Debit Card, UPI)",
        "category": "cod_refunds",
        "content": "Refunds for prepaid orders paid using UPI, Credit Cards, Debit Cards, or Net Banking are automatically credited back to the original payment source. UPI refunds reflect within 24 to 48 hours, whereas credit and debit card refunds take 3 to 7 working days depending on the card-issuing bank. No manual bank detail entry is required for prepaid cancellations."
    },
    {
        "doc_id": "DOC_13_DAMAGE_IN_TRANSIT_POLICY",
        "title": "Policy for Damaged, Missing, or Wrong Item Delivered",
        "category": "return_windows",
        "content": "If a customer receives an order with physical damage, missing components, or an incorrect item, the incident must be reported within 48 hours of delivery. Clear unboxing photos or videos highlighting the package condition and shipping label are required for investigation. Approved claims are prioritized for urgent replacement or 100% refund."
    },
    {
        "doc_id": "DOC_14_NON_SERVICEABLE_REVERSE_PICKUP",
        "title": "Self-Ship Return Process for Non-Serviceable Reverse Pickup Areas",
        "category": "reverse_pickup",
        "content": "If a customer's pin code does not support reverse-pickup courier coverage, Flipkart enables the self-ship option. Customers can courier the package via India Post Speed Post or registered courier to the designated return fulfillment center. Flipkart reimburses the shipping receipt cost up to INR 300 upon receiving proof of postage."
    }
]

# Ground Truth Answer Key for Task 10 Retrieval Evaluation (6 realistic test queries)
RETRIEVAL_EVAL_ANSWER_KEY = [
    {
        "query_id": "Q1",
        "query": "What is the return window for shoes and sandals?",
        "relevant_docs": ["DOC_02_FOOTWEAR_RETURN_WINDOW"]
    },
    {
        "query_id": "Q2",
        "query": "How many days does it take to get my money back for a Cash on Delivery order?",
        "relevant_docs": ["DOC_06_COD_REFUND_TIMELINE_BANK", "DOC_07_COD_REFUND_INSTANT_WALLET"]
    },
    {
        "query_id": "Q3",
        "query": "Can I return a damaged smartphone or laptop for a full refund?",
        "relevant_docs": ["DOC_03_ELECTRONICS_RETURN_WINDOW", "DOC_13_DAMAGE_IN_TRANSIT_POLICY"]
    },
    {
        "query_id": "Q4",
        "query": "What happens during reverse pickup doorstep verification?",
        "relevant_docs": ["DOC_10_REVERSE_PICKUP_DOORSTEP_QC", "DOC_11_REVERSE_PICKUP_ATTEMPTS_SLA"]
    },
    {
        "query_id": "Q5",
        "query": "How fast is delivery to metro cities like Delhi or Bangalore?",
        "relevant_docs": ["DOC_08_DELIVERY_SLA_METRO"]
    },
    {
        "query_id": "Q6",
        "query": "What should I do if reverse pickup is not available in my village pin code?",
        "relevant_docs": ["DOC_14_NON_SERVICEABLE_REVERSE_PICKUP"]
    }
]

def chunk_policy_documents(docs: List[Dict]) -> List[Dict]:
    """
    Sentence-wise chunking of policy documents.
    Every multi-sentence document produces multiple chunks while retaining parent document mapping.
    """
    chunks = []
    for doc in docs:
        doc_id = doc["doc_id"]
        title = doc["title"]
        category = doc["category"]
        content = doc["content"]
        
        # Split sentences by period followed by space
        sentences = [s.strip() for s in re.split(r'(?<=\.)\s+', content) if s.strip()]
        for s_idx, sentence in enumerate(sentences):
            chunks.append({
                "chunk_id": f"{doc_id}_C{s_idx+1:02d}",
                "doc_id": doc_id,
                "doc_title": title,
                "category": category,
                "text": sentence,
                "full_doc_content": content
            })
    return chunks

if __name__ == "__main__":
    chunks = chunk_policy_documents(POLICY_DOCUMENTS)
    print(f"Total Policy Documents: {len(POLICY_DOCUMENTS)}")
    print(f"Total Sentence Chunks: {len(chunks)}")
    print(f"Total Evaluation Queries: {len(RETRIEVAL_EVAL_ANSWER_KEY)}")
