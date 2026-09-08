import os

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)


def get_supabase_client() -> Client:
    """Get the Supabase client."""
    return supabase


def get_supabase_service_role_client() -> Client:
    """Get the Supabase service role client."""
    service_role_url: str = os.getenv("SUPABASE_URL")
    service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    return create_client(service_role_url, service_role_key)
