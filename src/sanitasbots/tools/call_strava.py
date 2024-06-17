from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field


class Route(BaseModel):
    name: str = Field(description="Name of the route")


@tool
def add_route(route: Route):
    """Allows the user to upload a new route to the Strava profile"""
    print("*** This is our new route: ", route)
