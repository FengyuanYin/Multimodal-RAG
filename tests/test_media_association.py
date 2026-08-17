from agentic_rag.memory.media_association import detect_references, resolve_reference


def test_detects_chinese_and_english_references():
    refs = detect_references("见图 2、Figure 3、Fig. 4 和表格5、Table 6。", "doc", 7)
    assert [(ref.media_type, ref.label) for ref in refs] == [
        ("image", "图2"), ("image", "图3"), ("image", "图4"),
        ("table", "表5"), ("table", "表6"),
    ]
    assert all(ref.page == 7 for ref in refs)


def test_exact_ambiguous_and_page_snapshot_resolution():
    ref = detect_references("图1", "doc", 2)[0]
    exact = {"id": "img", "doc_id": "doc", "type": "image", "page": 2, "label": "图1"}
    assert resolve_reference(ref, [exact]).resolution == "exact"

    duplicate = {**exact, "id": "img2"}
    assert resolve_reference(ref, [exact, duplicate]).resolution == "unresolved"

    snapshot = {"id": "page2", "doc_id": "doc", "type": "page_snapshot", "page": 2}
    decision = resolve_reference(ref, [snapshot])
    assert decision.media_id == "page2"
    assert decision.resolution == "page_match"
    assert decision.confidence < 0.5
