# Authentication

The Meridian Payments API supports two authentication methods: API keys and OAuth 2.0 client credentials.

## API keys
Secret keys start with `mk_live_` for production and `mk_test_` for the sandbox. Send the key in the `Authorization: Bearer <key>` header. Never embed secret keys in client-side code or share them with anyone, including Meridian support staff.

## Key rotation
When you rotate a key from the dashboard, the previous key stays valid for a 24-hour grace period so deployments can roll over without downtime. After the grace period the old key is rejected with error E1001.

## OAuth 2.0 client credentials
Request a token from `POST /v2/oauth/token` using your client ID and client secret. Access tokens expire after 3600 seconds (1 hour). Expired tokens are rejected with error E1002; request a new token instead of retrying the same one. Refresh tokens are not issued for the client credentials flow.
