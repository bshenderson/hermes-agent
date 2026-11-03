"""A linewise filler instruction must preserve the exact quotation protocol."""
from agent.compression_source_quotes import _prompt


def test_mixed_segment_instruction_skips_filler_per_line():
    source = '[USER]:\nSample filler.\nOwner: Mira. Do not retry or send.\n'
    prompt = _prompt(source)
    assert 'Evaluate each source line separately' in prompt
    assert 'Repetition elsewhere must not suppress' in prompt
    assert 'A mixed segment still requires quotations of its durable facts.' in prompt
    assert 'Skip repetitive filler and routine status rows without durable facts.' not in prompt
    assert prompt.split('<source>\n')[1].split('\n</source>\n')[0] == source
    assert 'Copy complete relevant sentences verbatim; do not paraphrase' in prompt
    assert 'If this segment contains no durable information return {"excerpts":[]}.' in prompt
