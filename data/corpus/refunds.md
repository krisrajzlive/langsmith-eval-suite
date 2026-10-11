# Refunds

## Refund window
Card payments can be refunded within 120 days of the original charge. After 120 days the refund request is rejected and the merchant must pay the customer through another channel.

## Partial refunds
Partial refunds are supported. You may issue multiple partial refunds against one payment as long as the total refunded never exceeds the original amount.

## Timing
Refunds are submitted to the card network immediately but typically appear on the customer's statement within 5 to 10 business days.

## Idempotency
Send an `Idempotency-Key` header on `POST /v2/refunds`. Keys are remembered for 24 hours; replaying the same key with a different body fails with error E2004.

## Fees
The processing fee charged on the original payment is not returned when a payment is refunded.
