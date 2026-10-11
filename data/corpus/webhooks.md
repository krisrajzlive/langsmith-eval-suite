# Webhooks

## Signature verification
Every webhook request carries a `Meridian-Signature` header containing an HMAC-SHA256 signature of the raw request body, computed with your endpoint's signing secret. Reject requests whose signature does not match; the API reports mismatches during test deliveries as error E3001.

## Delivery and retries
Your endpoint must respond with a 2xx status within 10 seconds. Failed deliveries are retried up to 8 times over 24 hours using exponential backoff. After the final attempt the event is marked as failed and remains available through `GET /v2/events` for 30 days.

## Event types
Common events include `payment.succeeded`, `payment.failed`, `refund.created`, `payout.paid` and `dispute.opened`. Events can be delivered more than once, so handlers must be idempotent and de-duplicate on the event `id`.
