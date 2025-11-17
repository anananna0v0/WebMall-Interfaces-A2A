"""
Interactive RAG system for testing product searches.
Allows users to interactively search for products using the flexible agent.
"""

import asyncio
import os
from typing import Optional
from dotenv import load_dotenv

from rag_agent import FlexibleRAGAgent
from elasticsearch_client import ElasticsearchRAGClient

load_dotenv()

# Configuration
URLS = {
    "URL_1": "https://webmall-1.informatik.uni-mannheim.de",
    "URL_2": "https://webmall-2.informatik.uni-mannheim.de", 
    "URL_3": "https://webmall-3.informatik.uni-mannheim.de",
    "URL_4": "https://webmall-4.informatik.uni-mannheim.de",
    "URL_5": "https://webmall-solution.informatik.uni-mannheim.de"
}


class InteractiveRAG:
    """Interactive interface for the flexible RAG system."""
    
    def __init__(self, model: str = "gpt-4", analyze_model: str = "gpt-3.5-turbo"):
        self.agent = FlexibleRAGAgent(model=model, analyze_model=analyze_model)
        self.es_client = ElasticsearchRAGClient()
        self.history = []
    
    def print_banner(self):
        """Print welcome banner."""
        print("\n" + "="*60)
        print("🔍 Interactive RAG Product Search")
        print("="*60)
        print("\nAvailable webshops:")
        for key, url in URLS.items():
            print(f"  - {key}: {url}")
        print("\nCommands:")
        print("  - Type your product search query")
        print("  - 'history' - Show search history")
        print("  - 'clear' - Clear search history")
        print("  - 'help' - Show this help")
        print("  - 'quit' or 'exit' - Exit the program")
        print("="*60 + "\n")
    
    def print_results(self, result: dict):
        """Print search results in a formatted way."""
        print(f"\n📊 Search Summary:")
        print(f"  - Total searches performed: {result['search_count']}")
        print(f"  - Documents analyzed: {result['analyzed_count']}")
        print(f"  - Agent iterations: {result['iterations']}")
        
        if result['search_history']:
            print(f"\n🔍 Search Queries:")
            for i, search in enumerate(result['search_history'], 1):
                print(f"  {i}. \"{search['query']}\" → {search['num_results']} results")
        
        print(f"\n📦 Found Products:")
        if result['final_urls'] and result['final_urls'] != ["Done"]:
            for i, url in enumerate(result['final_urls'], 1):
                print(f"  {i}. {url}")
        else:
            print("  No matching products found.")
        
        # Show top analyzed documents
        if result['analyzed_documents']:
            print(f"\n📋 Top Analyzed Products:")
            top_docs = sorted(
                result['analyzed_documents'],
                key=lambda x: x.get('relevance_score', 0),
                reverse=True
            )[:5]
            
            for i, doc in enumerate(top_docs, 1):
                print(f"\n  {i}. {doc['title']}")
                print(f"     URL: {doc['url']}")
                print(f"     Relevance: {doc.get('relevance_score', 0):.1f}/10")
                if doc.get('reasoning'):
                    print(f"     Notes: {doc['reasoning'][:100]}...")
    
    def show_history(self):
        """Show search history."""
        if not self.history:
            print("\n📜 No search history yet.")
            return
        
        print(f"\n📜 Search History ({len(self.history)} searches):")
        for i, (query, result) in enumerate(self.history, 1):
            print(f"\n{i}. Query: \"{query}\"")
            print(f"   Found: {len([u for u in result['final_urls'] if u != 'Done'])} products")
            print(f"   Searches: {result['search_count']}, Analyzed: {result['analyzed_count']}")
    
    async def search(self, query: str) -> dict:
        """Execute a search query."""
        print(f"\n🔄 Searching for: \"{query}\"")
        print("This may take a moment as the agent explores different search strategies...\n")
        
        try:
            result = await self.agent.run(query)
            self.history.append((query, result))
            return result
        except Exception as e:
            print(f"\n❌ Error during search: {str(e)}")
            return None
    
    async def run(self):
        """Run the interactive loop."""
        self.print_banner()
        
        while True:
            try:
                # Get user input
                user_input = input("\n🔍 Enter search query (or command): ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self.print_banner()
                
                elif user_input.lower() == 'history':
                    self.show_history()
                
                elif user_input.lower() == 'clear':
                    self.history = []
                    print("\n🗑️  History cleared.")
                
                elif user_input:
                    # Execute search
                    result = await self.search(user_input)
                    if result:
                        self.print_results(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
        
        # Cleanup
        await self.es_client.close()


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive RAG Product Search")
    parser.add_argument("--model", default="gpt-4", help="Main model to use (default: gpt-4)")
    parser.add_argument("--analyze-model", default="gpt-3.5-turbo", 
                       help="Model for document analysis (default: gpt-3.5-turbo)")
    
    args = parser.parse_args()
    
    # Create and run interactive interface
    interactive = InteractiveRAG(model=args.model, analyze_model=args.analyze_model)
    await interactive.run()


if __name__ == "__main__":
    # Example queries to try:
    print("\n💡 Example queries to try:")
    print("  - Find the cheapest gaming monitor with 144Hz and bigger than 27 inch")
    print("  - Show me all wireless keyboards under $50")
    print("  - Find gaming laptops with RTX graphics cards")
    print("  - Search for ergonomic office chairs with lumbar support")
    print("  - Find smartphones with 5G and at least 128GB storage\n")
    
    asyncio.run(main())