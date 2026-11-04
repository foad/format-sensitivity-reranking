# Demo case: upload-limit|B3|soft|rich|tagged2|labelled|w11

Question: **Is 50 MB still the cap on single uploads?**

Top k = 3. Expected answer: 200 MB per request (raised from 50 MB in release 3.4.0). Stale answer: 50 MB.

Mixed assignment (one per source system, shared by every arm): api_reference yaml, changelog json, wiki toml, tickets json.

This is a constructed scenario, selected to illustrate the failure.

## Summary

| Condition | Key passage rank | Margin | |
|---|---|---|---|
| Untrained, mixed formats | 5 | -1.25 | ✓ |
| Untrained, standardised to markdown | 5 | -0.30 | ✓ |
| lam0 (fine-tuned, rank loss only), mixed formats | 5 | -0.38 | ✓ |
| lam1 (fine-tuned with the invariance term), mixed formats | 3 | +0.29 | ✓ |

## Every candidate's rank

| passage | source | untrained, mixed | untrained, standardised | lam0, mixed | lam1, mixed |
|---|---|---|---|---|---|
| wiki-large-files | wiki | 1 | 1 | 1 | 1 |
| tk-4471 | tickets | 2 | 2 | 2 | 2 |
| ref-v3-uploads | api_reference | 3 | 4 | 3 | 4 |
| tk-4519 | tickets | 4 | 3 | 4 | 5 |
| **cl-3-4-0** | changelog | 5 | 5 | 5 | 3 |
| ref-v2-uploads | api_reference | 6 | 8 | 6 | 6 |
| ref-rate-limits | api_reference | 7 | 6 | 7 | 8 |
| tk-4555 | tickets | 8 | 9 | 8 | 7 |
| wiki-choosing-region | wiki | 9 | 11 | 9 | 10 |
| ref-v2-batch | api_reference | 10 | 10 | 11 | 12 |
| cl-3-3-0 | changelog | 11 | 7 | 12 | 9 |
| tk-4630 | tickets | 12 | 12 | 10 | 11 |
| wiki-residency-faq | wiki | 13 | 14 | 13 | 13 |
| ref-regions | api_reference | 14 | 13 | 14 | 14 |
| tk-4802 | tickets | 15 | 15 | 15 | 15 |

## Reported, not required

| Condition | Key passage rank | Margin |
|---|---|---|
| lam0 (rank only), standardised to markdown | 4 | -0.12 |
| lam1 (invariance), standardised to yaml | 3 | +0.18 |
| lam1 (invariance), standardised to json | 3 | +0.30 |
| lam1 (invariance), standardised to toml | 3 | +0.21 |
| lam1 (invariance), standardised to inline_kv | 3 | +0.15 |
| lam1 (invariance), standardised to markdown | 2 | +0.11 |

## JSON hold-out arms (trained without JSON)

| Condition | Key passage rank | Margin |
|---|---|---|
| JSON hold-out lam0, mixed | 5 | -0.64 |
| JSON hold-out lam0, standardised to markdown | 4 | -0.15 |
| JSON hold-out lam1, mixed | 3 | +0.25 |
| JSON hold-out lam1, standardised to markdown | 3 | +0.04 |

## What each condition passes on

### 1. Untrained, mixed formats

Key passage (cl-3-4-0) ranks 5: **cut**, 1.25 pool SD below the cut.

1. `wiki-large-files` (wiki, toml)

```
space = "Solutions Engineering"
owner_team = "solutions"
labels = "uploads, large-files"
last_modified = "2024-10-22"
## Handling large files

Customers ingesting scanned archives often hit the upload limit before anything else. A single POST to /v3/uploads tops out at 50 MB, so anything bigger needs splitting before it reaches us. We usually recommend that they split PDFs by chapter on their side, which has the side benefit of better search results later.

If splitting isn't an option, raise it with the account team and we can look at a bulk import through the jobs endpoint instead.
```

2. `tk-4471` (tickets, json)

```
{"ticket": "FH-4471", "product": "ingest-api", "priority": "normal", "status": "closed", "tags": "uploads, payload-size", "created": "2025-02-03"}
Customer: getting 413 when uplaoding a 60MB PDF to /v3/uploads, smaller files are fine. is there a way round this?

Agent (Priya): Hi, that's the payload cap, it's 50 MB per request at the moment. You can either split the file or use the new resumable upload sessions, which take the file in parts. Closing for now, reopen if you get stuck.
```

3. `ref-v3-uploads` (api_reference, yaml)

```
---
title: Uploads
api_version: 3
status: current
endpoint: POST /v3/uploads
owner: storage-team
last_updated: 2024-12-02
---
POST /v3/uploads

Creates a document from a single request body. Send the file as application/octet-stream and set X-Fernhollow-Filename. Maximum payload: 50 MB. Larger bodies are rejected with 413 Payload Too Large.

Returns 201 with the document id and a checksum. Uploads are processed asynchronously; poll GET /v3/documents/{id} for status.
```

### 2. Untrained, standardised to markdown

Key passage (cl-3-4-0) ranks 5: **cut**, 0.30 pool SD below the cut.

1. `wiki-large-files` (wiki, markdown)

```
**space**: Solutions Engineering
**owner_team**: solutions
**labels**: uploads, large-files
**last_modified**: 2024-10-22

## Handling large files

Customers ingesting scanned archives often hit the upload limit before anything else. A single POST to /v3/uploads tops out at 50 MB, so anything bigger needs splitting before it reaches us. We usually recommend that they split PDFs by chapter on their side, which has the side benefit of better search results later.

If splitting isn't an option, raise it with the account team and we can look at a bulk import through the jobs endpoint instead.
```

2. `tk-4471` (tickets, markdown)

```
**ticket**: FH-4471
**product**: ingest-api
**priority**: normal
**status**: closed
**tags**: uploads, payload-size
**created**: 2025-02-03

Customer: getting 413 when uplaoding a 60MB PDF to /v3/uploads, smaller files are fine. is there a way round this?

Agent (Priya): Hi, that's the payload cap, it's 50 MB per request at the moment. You can either split the file or use the new resumable upload sessions, which take the file in parts. Closing for now, reopen if you get stuck.
```

3. `tk-4519` (tickets, markdown)

```
**ticket**: FH-4519
**product**: ingest-api
**priority**: low
**status**: closed
**tags**: uploads, payload-size
**created**: 2025-03-14

Customer: whats the biggest file we can push through /v3/uploads in one go? want to size our exports

Agent (Priya): Hi! It's 50 MB per request. For anything bigger the upload sessions endpoint takes files in 16 MB parts, so there's no practical ceiling there.
```

### 3. lam0 (fine-tuned, rank loss only), mixed formats

Key passage (cl-3-4-0) ranks 5: **cut**, 0.38 pool SD below the cut.

1. `wiki-large-files` (wiki, toml)

```
space = "Solutions Engineering"
owner_team = "solutions"
labels = "uploads, large-files"
last_modified = "2024-10-22"
## Handling large files

Customers ingesting scanned archives often hit the upload limit before anything else. A single POST to /v3/uploads tops out at 50 MB, so anything bigger needs splitting before it reaches us. We usually recommend that they split PDFs by chapter on their side, which has the side benefit of better search results later.

If splitting isn't an option, raise it with the account team and we can look at a bulk import through the jobs endpoint instead.
```

2. `tk-4471` (tickets, json)

```
{"ticket": "FH-4471", "product": "ingest-api", "priority": "normal", "status": "closed", "tags": "uploads, payload-size", "created": "2025-02-03"}
Customer: getting 413 when uplaoding a 60MB PDF to /v3/uploads, smaller files are fine. is there a way round this?

Agent (Priya): Hi, that's the payload cap, it's 50 MB per request at the moment. You can either split the file or use the new resumable upload sessions, which take the file in parts. Closing for now, reopen if you get stuck.
```

3. `ref-v3-uploads` (api_reference, yaml)

```
---
title: Uploads
api_version: 3
status: current
endpoint: POST /v3/uploads
owner: storage-team
last_updated: 2024-12-02
---
POST /v3/uploads

Creates a document from a single request body. Send the file as application/octet-stream and set X-Fernhollow-Filename. Maximum payload: 50 MB. Larger bodies are rejected with 413 Payload Too Large.

Returns 201 with the document id and a checksum. Uploads are processed asynchronously; poll GET /v3/documents/{id} for status.
```

### 4. lam1 (fine-tuned with the invariance term), mixed formats

Key passage (cl-3-4-0) ranks 3: **kept**, 0.29 pool SD above the cut.

1. `wiki-large-files` (wiki, toml)

```
space = "Solutions Engineering"
owner_team = "solutions"
labels = "uploads, large-files"
last_modified = "2024-10-22"
## Handling large files

Customers ingesting scanned archives often hit the upload limit before anything else. A single POST to /v3/uploads tops out at 50 MB, so anything bigger needs splitting before it reaches us. We usually recommend that they split PDFs by chapter on their side, which has the side benefit of better search results later.

If splitting isn't an option, raise it with the account team and we can look at a bulk import through the jobs endpoint instead.
```

2. `tk-4471` (tickets, json)

```
{"ticket": "FH-4471", "product": "ingest-api", "priority": "normal", "status": "closed", "tags": "uploads, payload-size", "created": "2025-02-03"}
Customer: getting 413 when uplaoding a 60MB PDF to /v3/uploads, smaller files are fine. is there a way round this?

Agent (Priya): Hi, that's the payload cap, it's 50 MB per request at the moment. You can either split the file or use the new resumable upload sessions, which take the file in parts. Closing for now, reopen if you get stuck.
```

3. `cl-3-4-0` (changelog, json)

```
{"release": "3.4.0", "date": "2025-04-08", "component": "uploads", "type": "changed", "title": "Raise the upload body limit", "affects": "POST /v3/uploads, POST /v3/upload-sessions", "breaking": "false"}
- Single-request body ceiling raised from 50 MB to 200 MB on all plans.
- 413 responses now include max_bytes in the error body.
- Fixed checksums occasionally missing from 201 responses.
```

