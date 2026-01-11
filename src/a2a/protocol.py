from typing import List, Optional, Union, Any, Dict, TypedDict
from dataclasses import dataclass, field
import uuid

# --- Schema.org Data Models ---

class SchemaOffer(TypedDict):
    """
    Represents an offer for a product based on Schema.org.
    Used for price and availability information.
    """
    type: str  # Always "Offer"
    price: float
    priceCurrency: str
    url: str
    availability: Optional[str]

class SchemaProduct(TypedDict):
    """
    Represents a product entity based on Schema.org.
    """
    type: str  # Always "Product"
    name: str
    description: Optional[str]
    image: Optional[str]
    sku: Optional[str]
    offers: Union[SchemaOffer, List[SchemaOffer]]

# --- JSON-RPC 2.0 Base Structures ---

class JSONRPCBase(TypedDict):
    jsonrpc: str  # Must be "2.0"

class JSONRPCRequest(JSONRPCBase):
    """
    Standard JSON-RPC 2.0 Request structure sent by Coordinator.
    """
    method: str
    params: Dict[str, Any]
    id: Union[str, int]

class JSONRPCResponse(JSONRPCBase):
    """
    Standard JSON-RPC 2.0 Response structure sent by Store Agent.
    """
    result: Optional[Any]
    error: Optional[Dict[str, Any]]
    id: Union[str, int]

# --- Specific Method Parameters ---

class SearchParams(TypedDict):
    """
    Parameters for the 'search_product' method.
    """
    query: str
    max_results: Optional[int]
    filters: Optional[Dict[str, str]]  # e.g., {"@type": "Product"}

class AddToCartParams(TypedDict):
    """
    Parameters for the 'add_to_cart' method.
    """
    product_id: str
    quantity: int

class CheckoutParams(TypedDict):
    """
    Parameters for the 'checkout' method.
    Includes user information required by task_sets.json.
    """
    first_name: str
    last_name: str
    email: str
    address: str
    city: str
    postcode: str
    country: str
    payment_method: str  # e.g., "cod", "bacs"

# --- Utility Class for Creating RPC Messages ---

class A2AProtocol:
    """
    Utility class to generate standardized JSON-RPC messages for A2A communication.
    """
    
    @staticmethod
    def create_request(method: str, params: Dict[str, Any], request_id: Optional[str] = None) -> JSONRPCRequest:
        """
        Creates a JSON-RPC 2.0 request dictionary.
        """
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id or str(uuid.uuid4())
        }

    @staticmethod
    def create_success_response(result: Any, request_id: Union[str, int]) -> JSONRPCResponse:
        """
        Creates a JSON-RPC 2.0 success response.
        """
        return {
            "jsonrpc": "2.0",
            "result": result,
            "error": None,
            "id": request_id
        }

    @staticmethod
    def create_error_response(code: int, message: str, request_id: Union[str, int]) -> JSONRPCResponse:
        """
        Creates a JSON-RPC 2.0 error response.
        """
        return {
            "jsonrpc": "2.0",
            "result": None,
            "error": {"code": code, "message": message},
            "id": request_id
        }