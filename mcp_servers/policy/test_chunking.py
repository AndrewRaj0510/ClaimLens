from chunking import chunk_pages, embedding_text, is_heading, page_lines


def doc(*lines):
    return "\n".join(lines)


def lines(pages):
    return [line for _, line in page_lines(pages)]


def test_rejoins_words_split_across_lines():
    assert lines(["the Declara-\ntions page"]) == ["the Declarations", "page"]


def test_keeps_real_hyphens_when_split():
    assert lines(["a non-\nowned auto"]) == ["a non-owned", "auto"]


def test_rejoins_across_a_page_break():
    assert lines(["we will pay ex-\nPage 2 of 9", "penses incurred"]) == [
        "we will pay expenses",
        "incurred",
    ]


def test_removes_page_furniture():
    pages = [
        "Page 1 of 4 Copyright, Insurance Services Office, Inc.\nfirst",
        "14\nsecond",
        "A-30ME (08-11) Page 3 of 18\nthird",
    ]
    assert lines(pages) == ["first", "second", "third"]


def test_drops_lines_repeated_on_most_pages():
    pages = [f"FORM X-1 (2020)\nclause {i}" for i in range(5)]
    assert lines(pages) == [f"clause {i}" for i in range(5)]


def test_skips_contents_pages():
    toc = "\n".join(f"PART {p} .............. {i}" for i, p in enumerate("ABCDEF"))
    assert lines([toc, "PART A - LIABILITY\nbody"]) == ["PART A - LIABILITY", "body"]


def test_unknown_glyph_becomes_a_dash():
    assert lines(["PART IV�DAMAGE TO A VEHICLE"]) == ["PART IV - DAMAGE TO A VEHICLE"]


def test_heading_detection():
    assert is_heading("EXCLUSIONS")
    assert is_heading("1. NOTICE")
    assert is_heading("PART D - COVERAGE FOR DAMAGE TO YOUR AUTO")
    assert not is_heading("We will not pay for:")
    assert not is_heading("1. A total theft of your auto")
    assert not is_heading("PP 00")  # too few letters: form numbers aren't headings


def test_heading_path_has_three_levels():
    page = doc(
        "PART D - COVERAGE FOR DAMAGE TO YOUR AUTO",
        "INSURING AGREEMENT",
        "We will pay for direct and accidental loss.",
        "CONDITIONS",
        "1. NOTICE",
        "Tell us promptly.",
        "EXCLUSIONS",
        "We will not pay for wear and tear.",
    )
    sections = [c["section"] for c in chunk_pages([page])]
    assert sections == [
        "PART D - COVERAGE FOR DAMAGE TO YOUR AUTO > INSURING AGREEMENT",
        "PART D - COVERAGE FOR DAMAGE TO YOUR AUTO > CONDITIONS > 1. NOTICE",
        "PART D - COVERAGE FOR DAMAGE TO YOUR AUTO > EXCLUSIONS",
    ]


def test_joins_headings_split_over_two_lines():
    page = doc(
        "PART IV - DAMAGE",
        "INSURING AGREEMENT - CUSTOM PARTS",
        "OR EQUIPMENT COVERAGE",
        "body one",
        "EXCLUSIONS - READ CAREFULLY. IF AN EX-",
        "CLUSION APPLIES, NO COVER.",
        "body two",
    )
    sections = [c["section"] for c in chunk_pages([page])]
    assert sections == [
        "PART IV - DAMAGE > INSURING AGREEMENT - CUSTOM PARTS OR EQUIPMENT COVERAGE",
        "PART IV - DAMAGE > EXCLUSIONS - READ CAREFULLY",
    ]


def test_bare_section_takes_its_title_from_the_next_line():
    page = "SECTION III\nPhysical Damage Coverages\nDEFINITIONS\nbody"
    assert chunk_pages([page])[0]["section"] == (
        "SECTION III - PHYSICAL DAMAGE COVERAGES > DEFINITIONS"
    )


def test_long_sections_split_with_overlap_and_page_ranges():
    pages = [
        "EXCLUSIONS\n" + "\n".join(f"word{i} " * 10 for i in range(20)),
        "more text",
    ]
    chunks = chunk_pages(pages, max_words=50)
    assert all(len(c["text"].split()) <= 50 for c in chunks)
    assert len(chunks) > 1
    # The last two lines of one chunk open the next one.
    first, second = chunks[0]["text"].splitlines(), chunks[1]["text"].splitlines()
    assert first[-2:] == second[:2]
    assert chunks[0]["page_start"] == 1 and chunks[-1]["page_end"] == 2
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_embedding_text_includes_the_heading_path():
    chunk = {"section": "PART D > EXCLUSIONS", "text": "We will not pay for wear."}
    assert embedding_text(chunk).startswith("PART D > EXCLUSIONS\n")
