#!/usr/bin/env python3
import json
import csv
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src' / 'nlweb_mcp'))

# Load environment variables
load_dotenv()

# Import WooCommerce client
from woocommerce_client import WooCommerceClient

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# WebMall shop configurations with URL mapping
WEBMALL_SHOPS = {
    "webmall-1": {
        "url": "https://webmall-1.informatik.uni-mannheim.de",
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_1", ""),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_1", "")
    },
    "webmall-2": {
        "url": "https://webmall-2.informatik.uni-mannheim.de",
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_2", ""),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_2", "")
    },
    "webmall-3": {
        "url": "https://webmall-3.informatik.uni-mannheim.de",
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_3", ""),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_3", "")
    },
    "webmall-4": {
        "url": "https://webmall-4.informatik.uni-mannheim.de",
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_4", ""),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_4", "")
    }
}

# URL placeholder mapping for task_sets.json
URL_MAPPING = {
    "{{URL_1}}": "https://webmall-1.informatik.uni-mannheim.de",
    "{{URL_2}}": "https://webmall-2.informatik.uni-mannheim.de",
    "{{URL_3}}": "https://webmall-3.informatik.uni-mannheim.de",
    "{{URL_4}}": "https://webmall-4.informatik.uni-mannheim.de"
}

# Initial error categories
INITIAL_CATEGORIES = {
    "Incorrect_product_variant": {
        "description": "The correct product type but missing or different specifications",
        "examples": [
            "Antec C8 Gaming Case instead of Antec C8 ARGB Gaming Case",
            "iPhone 15 128GB instead of iPhone 15 256GB",
            "RTX 4070 instead of RTX 4070 Ti"
        ]
    },
    "Wrong_brand": {
        "description": "Product serves same purpose but from a different manufacturer",
        "examples": [
            "Samsung Galaxy S24 instead of iPhone 15",
            "AMD Ryzen 9 instead of Intel Core i9",
            "Corsair RAM instead of G.Skill RAM"
        ]
    },
    "Wrong_product_type": {
        "description": "Product is from a completely different category",
        "examples": [
            "Gaming mouse when searching for Gaming keyboard",
            "Monitor when searching for Graphics card",
            "Laptop when searching for Desktop PC"
        ]
    },
    "Partially_matching": {
        "description": "Product has some matching characteristics but differs in important aspects",
        "examples": [
            "AMD Ryzen 7 when searching for AMD Ryzen 9",
            "32-inch monitor when searching for 27-inch monitor",
            "Wireless headset when searching for Wired headset"
        ]
    },
    "Completely_different": {
        "description": "Product has no connection to the search query",
        "examples": [
            "Kitchen appliance when searching for Computer component",
            "Clothing when searching for Electronics",
            "Book when searching for Hardware"
        ]
    },
    "Not_cheapest_option": {
        "description": "Selected product is correct type but more expensive than the cheapest option",
        "examples": [
            "Samsung Galaxy S24 at $850 when cheaper option at $750 exists",
            "Corsair RAM at $120 when identical spec at $99 exists",
            "SSD 1TB at $150 when same model at $130 exists"
        ]
    },
    "Wrong_comparison_criteria": {
        "description": "Compared wrong attribute or misunderstood the comparison requirement",
        "examples": [
            "Selected largest capacity when asked for fastest speed",
            "Selected highest price when asked for best value",
            "Selected most features when asked for simplest option"
        ]
    }
}

def load_error_categories(filepath: str = "error_categories.json") -> Dict[str, Dict]:
    """Load error categories from JSON file, create if doesn't exist."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        # Create initial categories file
        with open(filepath, 'w') as f:
            json.dump(INITIAL_CATEGORIES, f, indent=2)
        return INITIAL_CATEGORIES

def save_error_categories(categories: Dict[str, Dict], filepath: str = "error_categories.json"):
    """Save error categories to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(categories, f, indent=2)

def load_task_ground_truth(filepath: str = "task_sets.json") -> Dict[str, Dict]:
    """Load ground truth answers from task_sets.json."""
    ground_truth = {}
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Ground truth comparison will be limited.")
        return ground_truth
    
    with open(filepath, 'r') as f:
        task_sets = json.load(f)
    
    # Process each task set
    for task_set in task_sets:
        for task in task_set.get('tasks', []):
            task_id = task.get('id')
            if not task_id:
                continue
            
            # Get correct answers and replace URL placeholders
            correct_answers = task.get('correct_answer', {}).get('answers', [])
            resolved_answers = []
            for answer in correct_answers:
                # Replace placeholders with actual URLs
                for placeholder, actual_url in URL_MAPPING.items():
                    answer = answer.replace(placeholder, actual_url)
                resolved_answers.append(answer)
            
            ground_truth[task_id] = {
                'task': task.get('task', ''),
                'category': task.get('category', ''),
                'correct_urls': resolved_answers,
                'motivation': task.get('motivation', '')
            }
    
    return ground_truth

def fetch_products_from_urls(urls: List[str], cache: Dict[str, Any]) -> List[Dict]:
    """Fetch product details for a list of URLs."""
    products = []
    
    for url in urls:
        # Extract shop and slug from URL
        shop_id, product_slug = extract_shop_and_slug_from_url(url)
        if not shop_id or not product_slug:
            continue
        
        # Check cache
        cache_key = f"{shop_id}:{product_slug}"
        if cache_key in cache:
            products.append(cache[cache_key])
        else:
            # Fetch product
            product = fetch_product_from_woocommerce(shop_id, product_slug)
            if product:
                cache[cache_key] = product
                products.append(product)
                time.sleep(0.3)  # Be nice to the API
    
    return products

def extract_shop_and_slug_from_url(url: str) -> Tuple[str, str]:
    """Extract shop identifier and product slug from URL."""
    # Pattern: https://webmall-X.informatik.uni-mannheim.de/product/SLUG/
    match = re.match(r'https://(webmall-\d+)\.informatik\.uni-mannheim\.de/product/([^/]+)', url)
    if match:
        return match.group(1), match.group(2)
    return None, None

def fetch_product_from_woocommerce(shop_id: str, product_slug: str) -> Optional[Dict]:
    """Fetch product details from WooCommerce API."""
    if shop_id not in WEBMALL_SHOPS:
        print(f"Unknown shop: {shop_id}")
        return None
    
    shop_config = WEBMALL_SHOPS[shop_id]
    
    try:
        # Initialize WooCommerce client
        wc_client = WooCommerceClient(
            base_url=shop_config["url"],
            consumer_key=shop_config["consumer_key"],
            consumer_secret=shop_config["consumer_secret"]
        )
        
        # Search for product by slug using the slug parameter
        products = wc_client.get_products(slug=product_slug, per_page=1)
        
        if products and len(products) > 0:
            return products[0]
        
        print(f"Product not found: {product_slug} in {shop_id}")
        return None
        
    except Exception as e:
        print(f"Error fetching product from {shop_id}: {e}")
        return None

def analyze_error_with_gpt5(task_description: str, product_data: Dict, 
                            categories: Dict[str, Dict], 
                            correct_products: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Use GPT-5 to analyze the error and categorize it."""
    
    # Prepare incorrect product information
    product_info = {
        "name": product_data.get('name', 'Unknown'),
        "description": product_data.get('description', ''),
        "short_description": product_data.get('short_description', ''),
        "price": product_data.get('price', ''),
        "categories": [cat.get('name', '') for cat in product_data.get('categories', [])],
        "tags": [tag.get('name', '') for tag in product_data.get('tags', [])]
    }
    
    # Prepare correct products information if available
    correct_products_info = []
    if correct_products:
        for cp in correct_products:
            correct_products_info.append({
                "name": cp.get('name', 'Unknown'),
                "price": cp.get('price', ''),
                "url": cp.get('permalink', '')
            })
    
    # Prepare categories for prompt
    categories_text = ""
    for cat_name, cat_info in categories.items():
        categories_text += f"\n- **{cat_name}**: {cat_info['description']}\n"
        categories_text += f"  Examples: {', '.join(cat_info['examples'][:2])}\n"
    
    # Build correct products section
    correct_products_text = ""
    if correct_products_info:
        correct_products_text = "\n\nCorrect Products (Ground Truth):\n"
        for i, cp in enumerate(correct_products_info, 1):
            correct_products_text += f"{i}. {cp['name']}\n"
            correct_products_text += f"   Price: {cp['price']}\n"
        
        # Determine if this is a comparison task
        if "cheapest" in task_description.lower() or "lowest price" in task_description.lower():
            correct_products_text += "\nNote: This is a price comparison task - the cheapest option(s) should be selected.\n"
        elif "fastest" in task_description.lower() or "highest speed" in task_description.lower():
            correct_products_text += "\nNote: This is a performance comparison task - the fastest option(s) should be selected.\n"
    
    prompt = f"""You are an expert at analyzing e-commerce search errors. 

Task/Query: {task_description}

Product Found (Incorrect):
- Name: {product_info['name']}
- Description: {product_info['short_description']}
- Categories: {', '.join(product_info['categories'])}
- Price: {product_info['price']}
{correct_products_text}

Existing Error Categories:
{categories_text}

Analyze this error by comparing the incorrect product with the correct products. Consider:
- Is it the wrong variant/model?
- For price comparison tasks: Was a more expensive option chosen?
- For specification tasks: Are key features missing?
- Is it a completely different product?

Provide:
1. Error Category: Use an existing category if it fits, OR create a new one if needed
2. Severity Level (0-3):
   - 0: Not an error (exact match)
   - 1: Minor issue, user would still be satisfied
   - 2: Different but somewhat related product
   - 3: Completely wrong product
3. Explanation: Brief note explaining your decision

If creating a new category, also provide:
- New category name (use underscores, e.g., Missing_specification)
- Description of the new category
- 2-3 example cases

Respond in JSON format:
{{
    "error_category": "category_name",
    "severity_level": 0-3,
    "explanation": "your explanation",
    "new_category": {{  // Only if creating new category
        "name": "New_Category_Name",
        "description": "Description of the category",
        "examples": ["Example 1", "Example 2"]
    }}
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-5",  
            messages=[
                {"role": "system", "content": "You are an expert e-commerce error analyst. Respond only in valid JSON."},
                {"role": "user", "content": prompt}
            ],
        )
        
        # Parse response
        result_text = response.choices[0].message.content
        # Clean up response if needed
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        return json.loads(result_text)
        
    except Exception as e:
        print(f"Error analyzing with GPT: {e}")
        return {
            "error_category": "Analysis_failed",
            "severity_level": -1,
            "explanation": f"Failed to analyze: {str(e)}"
        }

def process_errors(input_csv: str, output_csv: str, limit: Optional[int] = None):
    """Process errors from CSV file and generate enhanced analysis."""
    
    # Load existing categories
    categories = load_error_categories()
    
    # Load ground truth
    ground_truth = load_task_ground_truth()
    print(f"Loaded ground truth for {len(ground_truth)} tasks")
    
    # Read input CSV
    errors_to_process = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        print(f"CSV columns found: {fieldnames}")
        
        for row in reader:
            # Only process "additional" type errors
            if row.get('type') == 'additional':
                errors_to_process.append(row)
    
    print(f"Found {len(errors_to_process)} 'additional' errors to process")
    
    # Limit processing if requested
    if limit:
        errors_to_process = errors_to_process[:limit]
        print(f"Processing limited to {limit} errors")
    
    # Prepare output
    enhanced_errors = []
    cache = {}  # Cache for product fetches
    
    for i, error in enumerate(errors_to_process, 1):
        print(f"\nProcessing error {i}/{len(errors_to_process)}")
        print(f"  Task: {error['task_id']}")
        print(f"  URL: {error['url']}")
        
        # Extract shop and product slug
        shop_id, product_slug = extract_shop_and_slug_from_url(error['url'])
        
        if not shop_id or not product_slug:
            print(f"  Could not parse URL")
            enhanced_error = error.copy()
            enhanced_error.update({
                'error_category': 'URL_parse_failed',
                'severity_level': -1,
                'analysis_note': 'Could not extract shop and product from URL'
            })
            enhanced_errors.append(enhanced_error)
            continue
        
        # Fetch product (with caching)
        cache_key = f"{shop_id}:{product_slug}"
        if cache_key in cache:
            product_data = cache[cache_key]
        else:
            print(f"  Fetching product from {shop_id}...")
            product_data = fetch_product_from_woocommerce(shop_id, product_slug)
            cache[cache_key] = product_data
            time.sleep(0.5)  # Be nice to the API
        
        if not product_data:
            enhanced_error = error.copy()
            enhanced_error.update({
                'error_category': 'Product_not_found',
                'severity_level': -1,
                'analysis_note': 'Product could not be fetched from WooCommerce'
            })
            enhanced_errors.append(enhanced_error)
            continue
        
        # Get ground truth for this task if available
        task_id = error['task_id']
        task_ground_truth = ground_truth.get(task_id, {})
        correct_products = []
        
        if task_ground_truth:
            print(f"  Fetching ground truth products...")
            correct_urls = task_ground_truth.get('correct_urls', [])
            correct_products = fetch_products_from_urls(correct_urls, cache)
            print(f"  Found {len(correct_products)} correct products")
        
        # Analyze with GPT-5
        print(f"  Analyzing with GPT...")
        analysis = analyze_error_with_gpt5(
            error['task_description'],
            product_data,
            categories,
            correct_products
        )
        
        # Check if new category was created
        if 'new_category' in analysis and analysis['new_category']:
            new_cat = analysis['new_category']
            categories[new_cat['name']] = {
                'description': new_cat['description'],
                'examples': new_cat['examples']
            }
            save_error_categories(categories)
            print(f"  Created new category: {new_cat['name']}")
        
        # Add analysis to error
        enhanced_error = error.copy()
        enhanced_error.update({
            'error_category': analysis.get('error_category', 'Unknown'),
            'severity_level': analysis.get('severity_level', -1),
            'analysis_note': analysis.get('explanation', ''),
            'product_name': product_data.get('name', '')
        })
        enhanced_errors.append(enhanced_error)
        
        print(f"  Category: {enhanced_error['error_category']}")
        print(f"  Severity: {enhanced_error['severity_level']}")
    
    # Write output CSV
    if enhanced_errors:
        fieldnames = list(enhanced_errors[0].keys())
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enhanced_errors)
        
        print(f"\nResults saved to: {output_csv}")
        
        # Print summary
        print("\nAnalysis Summary:")
        category_counts = {}
        severity_counts = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}
        
        for error in enhanced_errors:
            cat = error.get('error_category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
            sev = error.get('severity_level', -1)
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        print("\nErrors by Category:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")
        
        print("\nErrors by Severity:")
        for sev in [0, 1, 2, 3, -1]:
            if severity_counts[sev] > 0:
                sev_name = {0: "Not an error", 1: "Minor", 2: "Different but related", 
                           3: "Completely wrong", -1: "Failed"}[sev]
                print(f"  Level {sev} ({sev_name}): {severity_counts[sev]}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze e-commerce search errors with AI')
    parser.add_argument('--input', default='error_analysis_*.csv', 
                       help='Input CSV file (supports wildcards)')
    parser.add_argument('--output', default=None,
                       help='Output CSV file (default: error_analysis_enhanced_TIMESTAMP.csv)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of errors to process (for testing)')
    
    args = parser.parse_args()
    
    # Find input file
    input_files = list(Path('.').glob(args.input))
    if not input_files:
        print(f"No files matching pattern: {args.input}")
        return
    
    # Use most recent file
    input_file = sorted(input_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"Using input file: {input_file}")
    
    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'error_analysis_enhanced_{timestamp}.csv'
    
    # Process errors
    process_errors(str(input_file), output_file, args.limit)

if __name__ == '__main__':
    main()