---
name: Agent-Interaction-Protocol
description: Reference documentation for agent communication patterns - not an executable agent
model: none
color: gray
---

### Communication Format
```python
class AgentMessage:
    def __init__(self, sender, recipient, message_type, payload):
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type  # 'request', 'response', 'status', 'error'
        self.payload = payload
        self.timestamp = datetime.now()
        self.correlation_id = str(uuid.uuid4())
        
class AgentResponse:
    def __init__(self, status, data, errors=None, warnings=None):
        self.status = status  # 'success', 'partial', 'failed'
        self.data = data
        self.errors = errors or []
        self.warnings = warnings or []
        self.metadata = {}
```

### Orchestration Example
```python
# Project Manager orchestrating a complete workflow
pm = ProjectManagerAgent()
requirements = pm.gather_requirements(user_input)

# Phase 1: Data Acquisition
data_task = AgentMessage(
    sender='project_manager',
    recipient='data_wrangler',
    message_type='request',
    payload={'task': 'acquire_data', 'specs': requirements['data']}
)
data_response = data_wrangler.handle(data_task)

# Phase 2: EDA (if data acquisition successful)
if data_response.status in ['success', 'partial']:
    eda_task = AgentMessage(
        sender='project_manager',
        recipient='data_scientist',
        message_type='request',
        payload={'task': 'perform_eda', 'data': data_response.data}
    )
    eda_response = data_scientist.handle(eda_task)

# Continue through remaining phases...
```