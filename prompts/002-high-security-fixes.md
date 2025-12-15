<objective>
Fix the 9 HIGH severity security vulnerabilities identified in the security audit.
These are serious issues that could lead to data breaches or system compromise.
TDD approach: Write failing tests first, then implement fixes.
Depends on: 001-critical-security-fixes.md must be completed first.
</objective>

<context>
This is a Telegram bot for statistical analysis and ML training.
Security audit: @analyses/security-audit.md
Project conventions: @CLAUDE.md

Prerequisites: Critical fixes (prompt 001) must be applied first.
</context>

<high_issues>

<issue_1>
<location>src/bot/telegram_bot.py:130-147</location>
<type>Sensitive Data Exposure</type>
<problem>API keys loaded from .env without validation; keys could be logged in error messages</problem>
<fix>
1. Add API key format validation on load (check prefix, length, character set)
2. Sanitize error messages to never include API keys
3. Create masked logging for sensitive values (show only last 4 chars)
</fix>
</issue_1>

<issue_2>
<location>src/generators/validator.py:32</location>
<type>Injection</type>
<problem>open() forbidden but pandas I/O not blocked, allowing file access via pd.read_csv</problem>
<fix>
Add to forbidden patterns:
- `pd.read_csv`, `pd.read_excel`, `pd.read_json`, `pd.read_parquet`
- `pd.to_csv`, `pd.to_excel`, `pd.to_json`, `pd.to_parquet`
- `DataFrame.to_*` methods
</fix>
</issue_2>

<issue_3>
<location>src/utils/sanitization.py:60-62</location>
<type>Broken Access Control</type>
<problem>Dangerous characters regex missing backtick and unicode attack vectors</problem>
<fix>
Expand regex to include: `< > & " ' \` $ { } [ ] ( ) ; | \ \n \r \t \x00-\x1f`
Add unicode normalization before checking
</fix>
</issue_3>

<issue_4>
<location>src/core/state_manager.py:828-833</location>
<type>Data Size Limit</type>
<problem>DataFrame validation only checks memory, not row/column counts - DoS via wide datasets</problem>
<fix>
Add limits:
- Max rows: 1,000,000
- Max columns: 1,000
- Reject datasets exceeding limits with clear error message
</fix>
</issue_4>

<issue_5>
<location>src/processors/data_loader.py:370-377</location>
<type>Path Traversal</type>
<problem>Filename validation misses Windows UNC paths (\\server\share) and URL encoding</problem>
<fix>
Add validation for:
- Windows UNC paths: reject if starts with `\\`
- URL encoded traversal: reject `%2f`, `%2e`, `%5c` patterns
- Null bytes: reject `%00` or `\x00`
</fix>
</issue_5>

<issue_6>
<location>src/worker/http_server.py:68-98</location>
<type>Information Disclosure</type>
<problem>Worker /worker endpoint serves Python code without authentication</problem>
<fix>
1. Implement token-based authentication for /worker endpoint
2. Token should be generated at startup and shared via env var
3. Require X-Worker-Token header for /worker requests
4. Return 401 Unauthorized for missing/invalid tokens
</fix>
</issue_6>

<issue_7>
<location>src/utils/password_validator.py:114</location>
<type>Broken Authentication</type>
<problem>Single static password for all users, stored unhashed</problem>
<fix>
1. Implement password hashing with bcrypt (cost factor 12)
2. Store hashed passwords, not plaintext
3. Use secrets.compare_digest for timing-safe comparison
4. Add password complexity requirements (min 8 chars, mixed case, numbers)
</fix>
</issue_7>

<issue_8>
<location>src/bot/telegram_bot.py:487-492</location>
<type>Security Misconfiguration</type>
<problem>Worker URLs from env vars not validated, allowing SSRF attacks</problem>
<fix>
1. Validate URL format before use
2. Whitelist allowed hosts/ports (localhost, 127.0.0.1, configured hosts)
3. Reject URLs with credentials, unusual ports, or non-http(s) schemes
</fix>
</issue_8>

<issue_9>
<location>src/core/state_manager.py:943-971</location>
<type>Insecure Deserialization</type>
<problem>Session persistence uses JSON without signing, allowing session hijacking</problem>
<fix>
1. Implement session signing with HMAC-SHA256
2. Use SESSION_SIGNING_KEY from environment
3. Sign session data on save, verify on load
4. Reject sessions with invalid signatures
</fix>
</issue_9>

</high_issues>

<implementation>

<tdd_workflow>
For EACH issue:
1. Read existing code to understand current implementation
2. Write failing tests that verify the security fix works
3. Implement the fix
4. Run tests to confirm they pass
5. Verify no existing functionality is broken
</tdd_workflow>

<test_locations>
- Create: `tests/unit/test_security_high.py`
- Tests should be runnable with: `pytest tests/unit/test_security_high.py -v`
</test_locations>

<implementation_order>
Recommended order (dependencies):
1. sanitization.py (foundational)
2. validator.py (uses sanitization)
3. password_validator.py (independent)
4. data_loader.py (path validation)
5. state_manager.py - size limits
6. state_manager.py - session signing
7. telegram_bot.py - API key validation
8. telegram_bot.py - URL validation
9. http_server.py - worker auth
</implementation_order>

<constraints>
- Do NOT break existing functionality
- All secrets via environment variables
- Use standard libraries where possible (hashlib, secrets, bcrypt)
- Maintain backward compatibility for existing sessions (migration path)
- Worker token should be auto-generated if not provided
</constraints>

</implementation>

<output>
Files to modify:
- `./src/utils/sanitization.py` - Expanded character validation
- `./src/generators/validator.py` - Block pandas I/O
- `./src/utils/password_validator.py` - Bcrypt hashing
- `./src/processors/data_loader.py` - Enhanced path validation
- `./src/core/state_manager.py` - Size limits + session signing
- `./src/bot/telegram_bot.py` - API key + URL validation
- `./src/worker/http_server.py` - Token authentication
- `./tests/unit/test_security_high.py` - Security tests
</output>

<verification>
Before declaring complete:
1. Run: `pytest tests/unit/test_security_high.py -v` - All tests pass
2. Run: `pytest tests/ --ignore=tests/unit/test_data_loader.py --ignore=tests/integration/test_data_loader_telegram.py -q` - No regressions
3. Verify pandas I/O methods are blocked in validator
4. Verify password hashing uses bcrypt
5. Verify session data is signed
6. Verify worker endpoint requires authentication
7. Verify URL validation rejects SSRF patterns
</verification>

<success_criteria>
- [ ] API keys validated on load and never logged in plaintext
- [ ] Pandas I/O methods blocked in script validator
- [ ] Shell metacharacters regex expanded (backtick, null, unicode)
- [ ] DataFrame size limits enforced (1M rows, 1K columns)
- [ ] Path traversal blocks UNC paths and URL encoding
- [ ] Worker endpoint requires token authentication
- [ ] Passwords hashed with bcrypt, compared timing-safe
- [ ] Worker URLs validated against whitelist
- [ ] Sessions signed with HMAC-SHA256
- [ ] All new tests pass
- [ ] No regressions in existing tests
</success_criteria>
