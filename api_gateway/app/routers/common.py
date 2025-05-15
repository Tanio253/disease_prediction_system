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
        logger.info(f"Headers Being Forwarded: {headers_to_forward}") # Log headers being sent

        # Read request body content
        body_content = await request.body() # Read the body ONCE

        # --- Start Enhanced Logging ---
        logger.info(f"Gateway Forwarding To: {full_target_url}")
        logger.info(f"Original Request Method: {request.method}")
        # logger.info(f"Original Request Headers: {dict(request.headers)}") # Already have this

        # headers_to_forward = ...
        # logger.info(f"Headers Being Forwarded: {headers_to_forward}") # Already have this

        logger.info(f"Length of body_content to forward: {len(body_content)}")
        if body_content:
            try:
                # Try to decode as JSON for logging if content type suggests it
                if "application/json" in request.headers.get("content-type", "").lower():
                    decoded_body_for_log = body_content.decode('utf-8')
                    logger.info(f"Decoded JSON body being forwarded: {decoded_body_for_log}")
                else:
                    logger.info(f"Raw body (first 200 bytes) being forwarded: {body_content[:200]}")
            except Exception as e_log_body:
                logger.warning(f"Could not decode/log body fully: {e_log_body}")
                logger.info(f"Raw body (first 200 bytes) being forwarded (fallback): {body_content[:200]}")
        else:
            logger.info("Request body is empty.")
        # --- End Enhanced Logging ---

        try:
            backend_response = await client.request(
                method=request.method,
                url=full_target_url,
                headers=headers_to_forward,
                content=body_content, # Use the body_content read once
                timeout=120.0
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