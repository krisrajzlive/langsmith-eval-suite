# Rate limits

## Default limits
Each API key is limited to 100 requests per second, with a burst allowance of 200 requests. Enterprise plans raise the sustained limit to 1,000 requests per second.

## Exceeded limits
When a limit is exceeded the API returns HTTP 429 with error code E4290 and a `Retry-After` header giving the number of seconds to wait. Clients should use exponential backoff with jitter.

## Batch endpoint
The `POST /v2/batch` endpoint accepts up to 50 operations per call. Each operation inside a batch counts as one request against the rate limit, so a full batch consumes 50 units of the per-second budget.
