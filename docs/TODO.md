# Adopt new TimeseriesFrame class and generally upgrade timeseries handling

Here's my terminology-based checklist, which I can reference by name later.

---

## ✅ git checkout -b restructure-portfolio-pkg

- [ ] Create a temporary branch to experiment without touching `main`.

---

## ✅ git mv timeseries.py portfolio/timeseries.py

- [ ] Move the `TimeseriesFrame` class into a proper subpackage while preserving Git history.

---

## ✅ `__init__.py`

- [ ] Add portfolio/__init__.py to declare portfolio/ as a package and expose what I want:
```
    from .timeseries import TimeseriesFrame
    __all__ = ["TimeseriesFrame"]
```

---

## ✅ "Avoid full breakage with conservative changes"

- [ ] Move only `TimeseriesFrame` first. Don't touch data_loader.py or others yet.

---

## ✅ "Use relative imports inside the package"

- [ ] In `portfolio/data_loader.py`, do:
```
    from .timeseries import TimeseriesFrame
```

## ✅ "Use absolute imports outside the package"

- [ ] In `main.py`, do:
```
    from portfolio import TimeseriesFrame
```

## ✅ Formalize `portfolio/` into a real package (with `__init__.py` and relative imports)

- [ ] Move related modules into a `portfolio/` subdirectory and make it a proper Python package using `__init.py`.

    This includes:
    * Create portfolio/ and move modules like timeseries.py, data_loader.py, and utils.py into it
    * Add portfolio/__init__.py to declare it a package
	* Replace absolute imports between internal modules with relative ones
	* Use top-level imports in external scripts
	* Ensure the project root is in PYTHONPATH or run scripts from root (e.g., python main.py)

