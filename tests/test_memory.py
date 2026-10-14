from langchain_core.messages import AIMessage, HumanMessage

from analyst.memory import new_record, records_block, split_turns, summary_messages, trim_history


def convo(n):
    msgs = []
    for i in range(n):
        msgs += [HumanMessage(f"q{i}"), AIMessage(f"a{i}")]
    return msgs


def test_within_window_untouched():
    calls = []
    msgs, summary = trim_history(convo(6), "", 6, lambda s, t: calls.append(t) or "x")
    assert len(msgs) == 12 and summary == "" and calls == []


def test_older_turns_fold_into_summary():
    seen = {}

    def summarise(previous, turns):
        seen["previous"], seen["turns"] = previous, [m.text for m in turns]
        return "new summary"

    msgs, summary = trim_history(convo(8), "old summary", 6, summarise)
    assert summary == "new summary"
    assert seen == {"previous": "old summary", "turns": ["q0", "a0", "q1", "a1"]}
    assert [m.text for m in msgs][:2] == ["q2", "a2"] and len(msgs) == 12


def test_split_turns_groups_by_user_message():
    turns = split_turns(convo(3))
    assert [[m.text for m in t] for t in turns] == [["q0", "a0"], ["q1", "a1"], ["q2", "a2"]]


def test_records_are_separate_from_summary():
    rec = new_record("revenue by month", {"result_id": "r1", "data_ts": "t", "columns": ["month", "revenue"], "row_count": 10}, "SELECT\n  1")
    block = records_block([rec])
    assert "r1" in block and "month, revenue" in block and "SELECT 1" in block


def test_summary_prompt_forbids_numbers():
    msgs = summary_messages("", convo(1))
    assert "Do not include any figures" in msgs[0].content
    assert "User: q0" in msgs[1].content
