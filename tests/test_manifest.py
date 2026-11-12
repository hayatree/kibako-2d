from kibako.manifest import validate

H1 = "a" * 64
H2 = "b" * 64


def rec(**over):
    base = {
        "sha256": H1,
        "path": "items/sword.png",
        "source": "kenney",
        "source_url": "https://kenney.nl/assets/example",
        "license": "CC0-1.0",
    }
    return {**base, **over}


def test_valid_manifest_has_no_errors():
    assert validate([rec(), rec(sha256=H2, path="items/shield.png")]) == []


def test_rejects_license_outside_allowlist():
    for lic in ("CC-BY-3.0", "CC-BY-SA-3.0", "GPL-3.0-only", "cc0-1.0"):
        errors = validate([rec(license=lic)])
        assert len(errors) == 1 and "not allowed" in errors[0], lic


def test_rejects_missing_fields():
    errors = validate([rec(source_url="")])
    assert errors == ["line 1: missing source_url"]


def test_rejects_duplicate_hash_and_path():
    errors = validate([rec(), rec()])
    assert any("duplicate sha256" in e for e in errors)
    assert any("duplicate path" in e for e in errors)


def test_rejects_malformed_hash():
    assert any("sha256" in e for e in validate([rec(sha256="xyz")]))
