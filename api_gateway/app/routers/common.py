import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse # For streaming body
import logging

logger = logging.getLogger(__name__)

async def forward_request(request: Request, target_url: str, target_path: str):
    """
    Forwards an incoming request to a target URL and path.
    """
    async with httpx.AsyncClient() as client:
        # Construct the full target URL
        # The target_path might already include query parameters from the original request.url.path
        # and request.url.query
        
        full_target_url = f"{target_url}{target_path}"
        if request.url.query:
            full_target_url += f"?{request.url.query}"

        logger.info(f"Forwarding {request.method} request from {request.client.host} path {request.url.path} to {full_target_url}")

        # Prepare headers, excluding 'host' as httpx will set it correctly
        headers_to_forward = {
            key: value for key, value in request.headers.items() if key.lower() != 'host'
        }

        # Read request body content
        body_content = await request.body()

        try:
            # Make the request to the backend service
            backend_response = await client.request(
                method=request.method,
                url=full_target_url,
                headers=headers_to_forward,
                content=body_content, # Send raw bytes
                timeout=300.0 # General timeout, adjust as needed
            )

            # Log backend response status
            logger.info(f"Backend service at {full_target_url} responded with status {backend_response.status_code}")

            # To stream the response body directly without loading it all into memory:
            # This is good for large files or long responses.
            # Need to filter out certain headers that shouldn't be passed back directly.
            excluded_headers = ["content-encoding", "content-length", "transfer-encoding", "connection"]
            response_headers = {
                k: v for k, v in backend_response.headers.items() if k.lower() not in excluded_headers
            }
            
            # Return a StreamingResponse to efficiently pass back the backend's response
            return StreamingResponse(
                content=backend_response.aiter_bytes(), # Stream the content
                status_code=backend_response.status_code,
                headers=response_headers,
                media_type=backend_response.headers.get("content-type")
            )

        except httpx.ConnectError as e:
            logger.error(f"Connection error forwarding to {full_target_url}: {e}")
            return Response(content=f"Error connecting to backend service: {target_url}", status_code=503) # Service Unavailable
        except httpx.ReadTimeout as e:
            logger.error(f"Read timeout forwarding to {full_target_url}: {e}")
            return Response(content=f"Backend service timed out: {target_url}", status_code=504) # Gateway Timeout
        except Exception as e:
            logger.error(f"Unexpected error during request forwarding to {full_target_url}: {e}", exc_info=True)
            return Response(content="Internal server error in API Gateway.", status_code=500)