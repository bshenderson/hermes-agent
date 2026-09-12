"""Opt-in complete-input compression with bounded map/reduce prompts.

UTF-8 bytes upper-bound Qwen byte-BPE tokens. This deliberately under-fills
individual requests rather than gambling on a chars/token ratio for code or
Unicode. Output and chat-template reserves are independent of input size.
"""


class CompressionBudgetError(ValueError):
    """A configured lossless pass cannot safely fit its prompt."""


def lossless_settings():
    from agent.auxiliary_client import _get_auxiliary_task_config
    config = _get_auxiliary_task_config('compression')
    if config.get('lossless_input') is not True:
        return None
    context = config.get('context_length')
    output = config.get('max_tokens', 12000)
    if type(context) is not int or type(output) is not int or context <= output + 4096 or output <= 0:
        raise CompressionBudgetError('lossless compression requires explicit context and output budgets')
    # This byte-BPE bound is intentionally limited to the governed Qwen routes.
    if config.get('model') not in {'local-compression-256k', 'local-compression-gpu1-256k'}:
        raise CompressionBudgetError('lossless byte budget requires a governed local compression route')
    return {'context_length': context, 'output_tokens': output, 'safety_tokens': 1024}


def summarize_bounded(source, build_prompt, call, *, context_length, output_tokens,
                      safety_tokens=1024, max_levels=4, build_chunk_prompt=None):
    budget = context_length - output_tokens - safety_tokens
    if min(budget, output_tokens, safety_tokens) <= 0:
        raise ValueError('invalid compression budget')
    chunk_prompt = build_chunk_prompt or build_prompt
    overhead = max(len(build_prompt('').encode('utf-8')), len(chunk_prompt('').encode('utf-8')))
    available = budget - overhead
    if available < 256:
        raise ValueError('compression instructions exceed input budget')
    def invoke(text, *, chunk=False):
        prompt = (chunk_prompt if chunk else build_prompt)(text)
        if len(prompt.encode('utf-8')) > budget:
            raise ValueError('compression prompt exceeds input budget')
        result = call(prompt)
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError('empty chunk summary')
        return result
    current = source
    for _ in range(max_levels):
        raw = current.encode('utf-8')
        if len(raw) <= available:
            return invoke(current)
        chunks = []
        start = 0
        while start < len(raw):
            end = min(start + available, len(raw))
            while end < len(raw) and raw[end] & 0xC0 == 0x80:
                end -= 1
            chunks.append(raw[start:end].decode('utf-8'))
            start = end
        summaries = [invoke(chunk, chunk=True) for chunk in chunks]
        reduced = '\n\n'.join(f'CHUNK {i + 1} SUMMARY:\n{text}' for i, text in enumerate(summaries))
        if len(reduced.encode('utf-8')) >= len(raw):
            raise RuntimeError('chunk summaries did not shrink; preserve original conversation')
        current = reduced
    raise RuntimeError('compression reduction depth exceeded; preserve original conversation')
