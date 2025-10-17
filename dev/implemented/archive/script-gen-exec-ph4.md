# Script Generator & Executor Phase 4 Integration Plan

## 🎯 Executive Summary

This document outlines the Phase 4 integration plan to bridge the fully implemented script generator/executor system (Phases 1-3) with the existing Telegram bot interface. Currently, the bot routes statistical requests directly to StatsEngine, bypassing the secure script generation capabilities entirely.

**Complexity**: Medium (3-4 hours implementation)
**Priority**: Critical - Required for utilizing the script generation system
**Impact**: Enables dynamic Python script generation and secure execution via Telegram

---

## 📊 Current Architecture Gap

### What's Implemented ✅
- **Script Generator**: Secure template-based Python script generation
- **Script Executor**: Sandboxed subprocess execution with resource limits
- **Template Library**: Production-ready templates for stats and ML operations
- **Security Layer**: 70+ forbidden patterns, input sanitization, resource monitoring
- **Telegram Bot**: Full statistical analysis via direct StatsEngine integration

### What's Missing ❌
- **Integration Bridge**: No connection between script system and Telegram bot
- **Command Interface**: No `/script` command or script generation triggers
- **Orchestrator Routing**: Orchestrator only routes to StatsEngine, not script pipeline
- **Parser Support**: Parser doesn't recognize script generation requests

### Impact of Gap
```
User Request: "Generate a Python script for correlation analysis"
Current Flow: Parser → Orchestrator → StatsEngine (hardcoded logic)
Desired Flow: Parser → Orchestrator → ScriptGenerator → ScriptExecutor → Results
```

---

## 🚀 Integration Architecture

### High-Level Flow
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Telegram      │───▶│  Message Handler │───▶│  Request Parser │
│   /script cmd   │    │   (handlers.py)  │    │   (parser.py)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Telegram       │◀───│ Result Formatter │◀───│ Task Definition │
│  Response       │    │ (formatter.py)   │    │ (task_type="script")│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Execution      │◀───│ Task Orchestrator│◀───│                 │
│  Results        │    │ (orchestrator.py)│    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Script Pipeline │
                        │ Generator+Executor│
                        └─────────────────┘
```

### Component Integration Points

#### 1. Parser Enhancement (`src/core/parser.py`)
**New Functionality:**
- Recognize script generation patterns: `/script`, "generate script", "create Python code"
- Extract template type from natural language: "descriptive", "correlation", "classifier"
- Return TaskDefinition with `task_type="script"`

**Implementation:**
```python
def _parse_script_request(self, text: str) -> Optional[TaskDefinition]:
    """Parse script generation requests."""
    script_patterns = [
        r'/script\s+(\w+)',
        r'generate\s+(?:a\s+)?script\s+(?:for\s+)?(\w+)',
        r'create\s+(?:a\s+)?(?:python\s+)?script\s+(?:for\s+)?(\w+)'
    ]

    for pattern in script_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            operation = match.group(1).lower()
            return TaskDefinition(
                task_type="script",
                operation=operation,
                parameters=self._extract_script_parameters(text)
            )
    return None
```

#### 2. Orchestrator Enhancement (`src/core/orchestrator.py`)
**New Functionality:**
- Import ScriptGenerator and ScriptExecutor
- Add script routing alongside existing stats/ml routing
- Handle script execution results and error formatting

**Implementation:**
```python
async def _execute_script_task(
    self,
    task: TaskDefinition,
    dataframe: pd.DataFrame
) -> Dict[str, Any]:
    """Execute script generation and execution pipeline."""

    # Generate secure script
    generator = ScriptGenerator()
    script = generator.generate_script(task)

    # Execute in sandbox
    executor = ScriptExecutor()
    config = SandboxConfig(
        timeout=30,
        memory_limit=2048,
        allow_network=False
    )

    # Prepare input data
    input_data = {
        'dataframe': dataframe.to_dict(),
        'parameters': task.parameters
    }

    # Execute script
    result = await executor.run_sandboxed(script, input_data, config)

    return {
        'success': result.success,
        'output': result.output,
        'script_hash': result.script_hash,
        'execution_time': result.execution_time,
        'memory_usage': result.memory_usage,
        'metadata': {
            'operation': task.operation,
            'template_used': f"{task.operation}.j2",
            'security_validated': True,
            'resource_limits': config.__dict__
        }
    }
```

#### 3. Script Handler Creation (`src/bot/script_handler.py`)
**New File - Core Functionality:**
- Handle `/script` command with template listing
- Process script generation requests with natural language parsing
- Display generated scripts to users (optional security feature)
- Format execution results for Telegram display

**Key Functions:**
```python
async def script_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /script command - show available templates."""

async def script_generation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle script generation requests from natural language."""

async def display_script_preview(update: Update, script: str) -> bool:
    """Show generated script to user before execution (optional)."""

async def format_script_results(result: Dict[str, Any]) -> str:
    """Format script execution results for Telegram."""
```

#### 4. Message Handler Integration (`src/bot/handlers.py`)
**Modifications:**
- Add `/script` command routing to script_handler
- Add script generation pattern recognition in message_handler
- Route script requests to orchestrator with `task_type="script"`

**Implementation:**
```python
# Add to message_handler function
if any(pattern in message_text.lower() for pattern in ['/script', 'generate script', 'create python']):
    # Route to script generation pipeline
    task = parser.parse_request(message_text, user_id, conversation_id)
    if task and task.task_type == "script":
        result = await orchestrator.execute_task(task, dataframe)
        response_message = script_handler.format_script_results(result)
```

---

## 🔧 Implementation Steps

### Step 1: Parser Enhancement (30 minutes)
1. **Add script pattern recognition** to `src/core/parser.py`
2. **Implement parameter extraction** for script templates
3. **Add test cases** for script parsing patterns
4. **Verify TaskDefinition creation** with `task_type="script"`

### Step 2: Orchestrator Integration (45 minutes)
1. **Import script components** (ScriptGenerator, ScriptExecutor)
2. **Add script routing logic** to execute_task method
3. **Implement _execute_script_task** with full pipeline
4. **Add error handling** for script generation/execution failures

### Step 3: Script Handler Creation (60 minutes)
1. **Create `src/bot/script_handler.py`** with core functions
2. **Implement `/script` command** with template listing
3. **Add script preview functionality** (security feature)
4. **Create result formatting** for Telegram display

### Step 4: Message Handler Integration (30 minutes)
1. **Add script command routing** to handlers.py
2. **Integrate script pattern detection** in message_handler
3. **Connect to orchestrator pipeline** for script tasks
4. **Add error handling** for script-specific failures

### Step 5: Testing & Validation (45 minutes)
1. **Create integration tests** in `tests/integration/test_script_telegram.py`
2. **Test all command patterns** and natural language variants
3. **Verify security validations** and resource limits
4. **Validate result formatting** and error handling

---

## 🧪 Testing Scenarios

### Basic Command Testing
```bash
# Start bot
python src/bot/telegram_bot.py

# Upload test data
[Send CSV file to bot]

# Test script commands
/script
/script descriptive
/script correlation
/script train_classifier
```

### Natural Language Testing
```
"Generate a script to calculate statistics"
"Create Python code for correlation analysis"
"Generate a script for my data analysis"
"Make a Python script to train a classifier"
```

### Security & Resource Testing
```
# Test resource limits
/script descriptive [with large dataset]

# Test security validations
[Verify no dangerous code in generated scripts]

# Test execution monitoring
[Check execution time and memory usage reporting]
```

### Integration Testing
```python
# Test script vs direct comparison
direct_result = stats_engine.calculate_descriptive_stats(data, params)
script_result = script_executor.run_sandboxed(generated_script, data, config)

assert direct_result == json.loads(script_result.output)['result']
```

---

## 📁 File Modifications Summary

### New Files
```
src/bot/script_handler.py           # Script command handlers and formatting
tests/integration/test_script_telegram.py  # Integration tests
```

### Modified Files
```
src/core/parser.py                  # Add script parsing patterns
src/core/orchestrator.py            # Add script routing and execution
src/bot/handlers.py                 # Add /script command and routing
```

### Integration Points
```
ScriptGenerator ←→ Orchestrator     # Script generation integration
ScriptExecutor  ←→ Orchestrator     # Secure execution integration
Parser          ←→ ScriptHandler    # Natural language understanding
Formatter       ←→ ScriptHandler    # Result presentation
```

---

## 🎯 Success Criteria

### Functional Requirements ✅
- [ ] Bot responds to `/script` command with template listing
- [ ] Natural language script requests are parsed correctly
- [ ] Scripts are generated using secure templates
- [ ] Execution happens in sandboxed environment with resource limits
- [ ] Results are formatted appropriately for Telegram
- [ ] Error handling provides helpful user feedback

### Security Requirements ✅
- [ ] All generated scripts pass security validation (70+ forbidden patterns)
- [ ] Input sanitization prevents injection attacks
- [ ] Resource limits prevent system abuse
- [ ] Script execution is properly isolated
- [ ] No dangerous code execution possible

### Performance Requirements ✅
- [ ] Script generation completes in <10ms
- [ ] Execution respects timeout limits (default: 30s)
- [ ] Memory usage stays within limits (default: 2GB)
- [ ] Results match direct StatsEngine output accuracy

### User Experience Requirements ✅
- [ ] Clear command interface with helpful examples
- [ ] Informative error messages with suggestions
- [ ] Execution progress indication for long operations
- [ ] Resource usage reporting for transparency

---

## 🚨 Risk Assessment

### Technical Risks
- **Script Generation Errors**: Template parsing or parameter injection failures
  - *Mitigation*: Comprehensive error handling and fallback to StatsEngine
- **Execution Timeouts**: Long-running scripts exceeding limits
  - *Mitigation*: Configurable timeouts and user feedback
- **Resource Exhaustion**: Scripts consuming excessive memory/CPU
  - *Mitigation*: Robust resource monitoring and enforcement

### Security Risks
- **Template Injection**: Malicious parameters in script templates
  - *Mitigation*: Input sanitization and parameter validation
- **Sandbox Escape**: Scripts bypassing security restrictions
  - *Mitigation*: Multiple security layers and pattern validation
- **Resource Abuse**: Users triggering expensive operations
  - *Mitigation*: Rate limiting and resource quotas

### Integration Risks
- **Orchestrator Complexity**: Adding script routing increases complexity
  - *Mitigation*: Clean separation of concerns and comprehensive testing
- **Parser Conflicts**: Script patterns conflicting with existing parsing
  - *Mitigation*: Specific pattern ordering and priority handling

---

## 🔮 Future Enhancements

### Phase 5: Advanced Features
- **Custom Templates**: User-defined script templates
- **Script Sharing**: Share generated scripts between users
- **Script History**: Track and reuse previous generations
- **Batch Execution**: Process multiple datasets with same script

### Phase 6: Production Features
- **Rate Limiting**: Prevent script generation abuse
- **Usage Analytics**: Track template usage and performance
- **User Quotas**: Limit resource consumption per user
- **Caching**: Cache script results for repeated operations

### Integration Opportunities
- **ML Pipeline**: Full ML workflow with script generation
- **Visualization**: Generate scripts that create charts/plots
- **Export**: Scripts that save results to various formats
- **Scheduling**: Periodic script execution with cron-like interface

---

## 📚 References

### Related Documentation
- [Script Generator Implementation](./README.md#script-generator-and-executor-system)
- [Security Architecture](./README.md#security-architecture)
- [Template System](./README.md#template-system-architecture)
- [Orchestrator Enhanced](./README.md#enhanced-orchestrator-implementation)

### Code References
```
src/generators/script_generator.py:158   # Main script generation logic
src/execution/executor.py:287           # Secure execution implementation
src/generators/validator.py:225         # Security validation system
templates/stats/descriptive.j2:197      # Example statistical template
src/core/orchestrator.py:845           # Current orchestrator implementation
```

### Testing References
```
tests/unit/test_script_generator.py     # Script generation tests
tests/unit/test_script_executor.py      # Execution tests
tests/unit/test_script_validator.py     # Security validation tests
```

---

**Document Version**: 1.0
**Last Updated**: January 2025
**Implementation Priority**: Critical
**Estimated Effort**: 3-4 hours
**Dependencies**: Phases 1-3 (completed)