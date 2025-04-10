"""
test_discus_cage.py

Test script to verify our improved ranking and LLM responses for technical information.
"""

import os
import logging
from dotenv import load_dotenv
from app_rag import rank_docs_by_relevance
from app_llm import get_answer
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

def test_discus_cage_ranking():
    """
    Test our new ranking system with discus cage information
    """
    print("=== Testing Improved Ranking for Technical Information ===")
    
    # Create sample documents including discus cage and irrelevant information
    sample_docs = [
        Document(
            page_content="Introduction to Athletics and Field Events. This document covers various aspects of athletics competitions.",
            metadata={"source": "Competition Rules.pdf", "page": 1}
        ),
        Document(
            page_content="Discus Cage. All discus throws shall be made from an enclosure or cage to ensure the safety of spectators, officials and athletes.",
            metadata={"source": "Competition Rules.pdf", "page": 35}
        ),
        Document(
            page_content="The cage should be U-shaped in plan as shown in Figure TR35. The width of the mouth should be 6m, positioned 7m in front of the centre of the throwing circle.",
            metadata={"source": "Competition Rules.pdf", "page": 36}
        ),
        Document(
            page_content="General competition rules and regulations for World Athletics events.",
            metadata={"source": "Competition Rules.pdf", "page": 3}
        ),
        Document(
            page_content="The cage specified in this Rule is intended for use when the event takes place in the Field of Play with other events taking place at the same time or when the event takes place outside the Field of Play with spectators present.",
            metadata={"source": "Competition Rules.pdf", "page": 35}
        ),
    ]
    
    # Test ranking with discus cage query
    query = "discus throw cage"
    
    # Rank documents
    ranked_docs = rank_docs_by_relevance(sample_docs, query)
    
    # Display results
    print("\nRanked Documents (from most to least relevant):")
    for i, doc in enumerate(ranked_docs):
        print(f"\n{i+1}. Document from {doc.metadata['source']} (Page {doc.metadata['page']}):")
        print(f"   Content: {doc.page_content}")
    
    # Test LLM response
    sample_context = """From Competition Rules.pdf (Page 35):
Discus Cage. All discus throws shall be made from an enclosure or cage to ensure the safety of spectators, officials and athletes. The cage specified in this Rule is intended for use when the event takes place in the Field of Play with other events taking place at the same time or when the event takes place outside the Field of Play with spectators present.

From Competition Rules.pdf (Page 36):
The cage should be U-shaped in plan as shown in Figure TR35. The width of the mouth should be 6m, positioned 7m in front of the centre of the throwing circle. The height of the netting panels or draped netting at their lowest point should be at least 4m and it should be at least 6m for the 3m nearest the front of the cage on each side.

From Competition Rules.pdf (Page 35):
The cage should be designed, manufactured and maintained so as to be capable of stopping a 2kg discus moving at a speed of up to 25 metres per second. The arrangement should be such that there is no danger of ricocheting or rebounding back towards the athlete or over the top of the cage. Provided that it satisfies all the requirements of this Rule, any form of cage design and construction can be used.
"""
    
    print("\n\n=== Testing LLM Response with Discus Cage Information ===")
    print("\nGenerating answer...")
    try:
        answer = get_answer("tell me about discus throw cage", sample_context)
        print("\nAnswer received:")
        print("-" * 50)
        print(answer)
        print("-" * 50)
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_discus_cage_ranking() 