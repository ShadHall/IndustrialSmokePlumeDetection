def test_package_imports():
    """Smoke-level: the package must be importable."""
    import smoke_detection

    assert smoke_detection.__version__
