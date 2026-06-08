"""The vendored theorem registry loads and lists theorem specs."""
def test_registry_nonempty():
    from theory.theorem_registry import list_theorems
    specs = list_theorems()
    assert isinstance(specs, list) and len(specs) > 0
