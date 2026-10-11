# Security and compliance

## Certifications
Meridian is certified as a PCI DSS Level 1 service provider.

## Transport security
All API traffic requires TLS 1.2 or higher. Requests over plain HTTP or older TLS versions are refused.

## IP allowlisting
Dashboard and API access can be restricted to an allowlist of IP addresses, configured under Settings > Security.

## Data handling
Never log full card numbers, CVC values, or secret API keys. Meridian retains transaction records for 7 years to meet financial regulatory requirements. Meridian support will never ask for your secret API key or signing secret.
