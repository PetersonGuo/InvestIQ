# Python migration reference

This is the previous Python backend and its tests, retained for comparison during the C++ migration. Nothing here is imported or launched by the active app. The maintained backend is `backend/native/`; current tests are in `backend/tests/native/`.

The standalone reference accounting model used by current parity tests is `backend/tests/reference_engine.py`. This snapshot preserves the earlier API, data adapters, and alert behavior for inspection. It does not contain the live `.env` or database.
