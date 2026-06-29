from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

# Set up logger for capturing exceptions
logger = logging.getLogger(__name__)

# Function to add custom error handlers to the FastAPI app
def add_error_handlers(app: FastAPI):
    
    # General exception handler for all unhandled exceptions
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        # Log the unhandled exception with its traceback for debugging
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        # Return a JSON response with a 500 status code indicating a server error
        return JSONResponse(
            status_code=500,
            content={"message": "An unexpected error occurred. Please try again later."}
        )

    # Custom handler for ValueError exceptions
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        # Log the ValueError with a warning level (as it's a client-side issue)
        logger.warning(f"ValueError: {str(exc)}")
        # Return a JSON response with a 400 status code indicating a bad request
        return JSONResponse(
            status_code=400,
            content={"message": str(exc)}  # Return the exception message to the client
        )
