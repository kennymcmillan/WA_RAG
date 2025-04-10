"""
test_llm_prompt.py

Test script to validate that our LLM prompt correctly formats and passes context.
"""

import os
import logging
from dotenv import load_dotenv
from app_llm import get_answer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

def test_llm_prompt():
    """
    Test that our LLM prompt formatting is working correctly
    """
    print("=== Testing LLM Prompt Formatting ===")
    
    # Create a sample context with information about a person
    sample_context = """From Sanguine Sports Tech Fund.pdf (Page 1):
Sanguine AI Sports and Health Tech Fund Proposal by: Kenny McMillan PhD Brian Moore PhD 1. Sports Tech overview Sports technology (sports tech) represents a rapidly evolving sector that integrates innovation and digital solutions into athletic training, performance analysis, and sports management.

From Sanguine Sports Tech Fund.pdf (Page 2):
Dr. Brian Moore is an expert in AI applications for sports analytics. He has pioneered several breakthrough technologies in motion capture and performance optimization. His previous venture, SportMetrics AI, was acquired by a major sports equipment manufacturer.

From Sanguine Sports Tech Fund.pdf (Page 3):
The founding team combines Kenny McMillan's expertise in sports physiology with Brian Moore's background in artificial intelligence and computer vision. Together, they have over 30 years of experience in sports technology development.
"""
    
    # Test queries
    test_cases = [
        {
            "query": "Who is Brian Moore?",
            "description": "Testing query about a person"
        },
        {
            "query": "What is sports technology?",
            "description": "Testing query about a concept"
        }
    ]
    
    for test_case in test_cases:
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\n\n=== Test: {description} ===")
        print(f"Query: '{query}'")
        
        # Generate answer using the updated prompt
        print("\nGenerating answer...")
        try:
            answer = get_answer(query, sample_context)
            print("\nAnswer received:")
            print("-" * 50)
            print(answer)
            print("-" * 50)
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_llm_prompt() 