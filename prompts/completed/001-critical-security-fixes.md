<objective>
Fix the 3 CRITICAL security vulnerabilities identified in the security audit.
These are the highest-risk issues that must be resolved before production deployment.
TDD approach: Write failing tests first, then implement fixes.
</objective>

<context>
This is a Telegram bot for statistical analysis and ML training.
Security audit: @analyses/security-audit.md
Project conventions: @CLAUDE.md

The 3 critical vulnerabilities are:
1. **Hardcoded Secrets** (config/config.yaml:92) - FILE_PATH_PASSWORD in plaintext
2. **Security Misconfiguration** (src/execution/executor.py:224-227) - Resource limits disabled
3. **Insecure Deserialization** (src/engines/model_manager.py:save_model) - Pickle without signing
</context>

<critical_issues>

<issue_1>
<location>config/config.yaml:92</location>
<type>Hardcoded Secrets</type>
<problem>Password stored in plaintext configuration file (FILE_PATH_PASSWORD), providing file system access to anyone with config access</problem>
<fix>
1. Remove password from config.yaml entirely
2. Load from environment variable only
3. Update any code that reads this config value to use os.environ
</fix>
</issue_1>

<issue_2>
<location>src/execution/executor.py:224-227</location>
<type>Security Misconfiguration</type>
<problem>Resource limits and sandbox environment disabled in production (commented out), allowing unlimited memory/CPU usage and environment variable access</problem>
<fix>
1. Uncomment and enable resource limits for production
2. Add configuration flag to control sandbox mode (enabled by default)
3. Implement memory limit (2GB), CPU time limit (30s), process limit
</fix>
</issue_2>

<issue_3>
<location>src/engines/model_manager.py (save_model/load_model)</location>
<type>Insecure Deserialization</type>
<problem>Model persistence uses pickle serialization without signature verification, allowing arbitrary code execution if attacker controls model files</problem>
<fix>
1. Implement HMAC-SHA256 signing for model files
2. Generate signing key from environment variable (MODEL_SIGNING_KEY)
3. Sign models on save, verify signature on load
4. Reject models with invalid/missing signatures
</fix>
</issue_3>

</critical_issues>

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
- Create: `tests/unit/test_security_critical.py`
- Tests should be runnable with: `pytest tests/unit/test_security_critical.py -v`
</test_locations>

<constraints>
- Do NOT break existing functionality (run existing tests after changes)
- Use environment variables for ALL secrets (never hardcode)
- Signing key should be configurable but have secure default behavior (reject unsigned)
- Resource limits must be configurable for development vs production
</constraints>

</implementation>

<output>
Files to modify/create:
- `./config/config.yaml` - Remove hardcoded password
- `./src/execution/executor.py` - Enable resource limits
- `./src/engines/model_manager.py` - Add HMAC signing
- `./src/utils/model_signing.py` - New utility for model signing (if needed)
- `./tests/unit/test_security_critical.py` - Security tests
</output>

<verification>
Before declaring complete:
1. Run: `pytest tests/unit/test_security_critical.py -v` - All tests pass
2. Run: `pytest tests/ --ignore=tests/unit/test_data_loader.py --ignore=tests/integration/test_data_loader_telegram.py -q` - No regressions
3. Verify config.yaml has NO plaintext passwords
4. Verify executor.py has resource limits enabled by default
5. Verify model files are signed on save and verified on load
</verification>

<success_criteria>
- [ ] No passwords in config.yaml
- [ ] Resource limits active in executor (memory ≤2GB, CPU ≤30s)
- [ ] Model files are HMAC-signed
- [ ] Invalid/unsigned models are rejected on load
- [ ] All new tests pass
- [ ] No regressions in existing tests
</success_criteria>
