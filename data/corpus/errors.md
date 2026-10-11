# Error codes

| Code | Name | Meaning |
| --- | --- | --- |
| E1001 | invalid_api_key | The API key is missing, malformed, revoked, or past its rotation grace period. |
| E1002 | expired_token | The OAuth access token has expired. Request a new token. |
| E2001 | insufficient_funds | The balance is too low to cover the requested refund or payout. |
| E2004 | duplicate_idempotency_key | An Idempotency-Key was reused with a different request body. |
| E3001 | webhook_signature_mismatch | The webhook signature could not be verified against the signing secret. |
| E4290 | rate_limited | The per-key rate limit was exceeded. Honor the Retry-After header. |

All errors return a JSON body with `code`, `name`, `message` and a `request_id`. Include the `request_id` when contacting support.
