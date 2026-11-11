# TraceLine Integration & Support Guide

> FICTIONAL SAMPLE DATA for pipeline testing. Replace with your own documents.

## API basics

The TraceLine API base URL is `https://api.traceline.example/v2`. Authentication uses the OAuth2 client-credentials flow. Access tokens expire after 3600 seconds and must be refreshed by requesting a new token.

### Rate limits

Each API key is limited to 600 requests per minute on the Standard plan and 3,000 requests per minute on the Enterprise plan. Requests beyond the limit receive HTTP 429 with a `Retry-After` header indicating how many seconds to wait.

### Webhooks

TraceLine can push the events `shipment.created`, `shipment.delayed` and `shipment.delivered` to a customer endpoint. Failed deliveries are retried up to 8 times with exponential backoff over roughly 24 hours. Every webhook is signed with HMAC-SHA256; the signature is sent in the `X-TraceLine-Signature` header and should be verified with the shared webhook secret.

## Error codes

- **TL-1001** — Invalid or expired access token. Request a new token and retry.
- **TL-1042** — Device not paired. The ELD or telematics device ID has not been linked to a vehicle. Pair it under Settings > Devices.
- **TL-2210** — Geofence overlap. Two geofences for the same facility overlap by more than 15%. Merge or resize them.
- **TL-3307** — Telematics feed stale. No position update has been received for more than 30 minutes. Check the device's connectivity and power.
- **TL-4500** — Internal error. Contact support and include the request ID from the response.

## Data retention and residency

Position history is retained for 13 months on the Standard plan and 36 months on the Enterprise plan. Data for EU customers is stored exclusively in the Frankfurt (eu-central) region.

## Support tiers

Standard customers receive email support with a first response within 8 business hours. Enterprise customers receive 24/7 phone support, a 1-hour first response for Severity 1 incidents, and a dedicated customer success manager.
