import verifiers.v1 as vf

from ragent_core.retrievers.tool_protocol import ToolName, output_evidence_document_ids


def _message_text(message: vf.ToolMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        part.text for part in content if isinstance(part, vf.TextContentPart)
    )


def evidence_document_ids(trace: vf.Trace) -> set[int]:
    evidence_calls = {
        call.id: call.name
        for message in trace.assistant_messages
        for call in message.tool_calls or []
        if call.name in {ToolName.READ, ToolName.SEARCH}
    }
    evidence_ids: set[int] = set()
    for message in trace.tool_messages:
        tool_name = evidence_calls.get(message.tool_call_id)
        if tool_name is None:
            continue
        content = _message_text(message)
        evidence_ids.update(output_evidence_document_ids(tool_name, content))
    return evidence_ids
