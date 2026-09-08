// Loaded on demand only for user-authored Python strategies.
#include "build_config.hpp"
#include "strategy_api.h"
#include <Python.h>
#include <cmath>
#include <fstream>
#include <string>
using Callback = double (*)(const SA_Bar *, int, double, int64_t, const char *);
namespace {
PyObject *callback = nullptr;
PyObject *globals = nullptr;
double python_on_bar(const SA_Bar *bars, int count, double cash, int64_t shares, const char *) {
  PyObject *result = PyObject_CallFunction(callback, "idLd", count, cash, (long long)shares,
                                           cash + shares * bars[count - 1].close);
  if (!result) {
    PyErr_Print();
    return NAN;
  }
  double value = -1;
  if (result != Py_None) {
    if (PyBool_Check(result) || (!PyFloat_Check(result) && !PyLong_Check(result)))
      value = NAN;
    else {
      value = PyFloat_AsDouble(result);
      if (value < 0 || value > 1 || PyErr_Occurred()) {
        PyErr_Print();
        value = NAN;
      }
    }
  }
  Py_DECREF(result);
  return value;
}
} // namespace
bool load_python(const char *source, const char *bars, const char *params, const char *loader) {
  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  config.use_environment = 0;
  config.write_bytecode = 0;
  PyConfig_SetBytesString(&config, &config.program_name, SA_PYTHON_EXECUTABLE);
  auto status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status))
    return false;
  std::ifstream file(std::string(SA_BACKEND_ROOT) + "/simulation/python_strategy.py");
  std::string code{std::istreambuf_iterator<char>(file), {}};
  if (code.empty())
    return false;
  globals = PyDict_New();
  PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins());
  PyObject *result = PyRun_String(code.c_str(), Py_file_input, globals, globals);
  if (!result) {
    PyErr_Print();
    return false;
  }
  Py_DECREF(result);
  callback =
      PyObject_CallFunction(PyDict_GetItemString(globals, loader), "sss", source, bars, params);
  if (!callback) {
    PyErr_Print();
    return false;
  }
  return true;
}
extern "C" Callback sa_python_load(const char *source, const char *bars, const char *params) {
  return load_python(source, bars, params, "load") ? python_on_bar : nullptr;
}
using PortfolioCallback = int (*)(const SA_Asset *, int, int, double, double, const char *,
                                  double *);
int python_on_portfolio(const SA_Asset *assets, int n, int count, double cash, double equity,
                        const char *, double *weights) {
  PyObject *positions = PyTuple_New(n);
  for (int a = 0; a < n; ++a)
    PyTuple_SET_ITEM(positions, a, PyLong_FromLongLong(assets[a].shares));
  PyObject *result = PyObject_CallFunction(callback, "iddO", count, cash, equity, positions);
  Py_DECREF(positions);
  if (!result) {
    PyErr_Print();
    return -1;
  }
  if (result == Py_None) {
    Py_DECREF(result);
    return 0;
  }
  if (!PyList_Check(result) || PyList_Size(result) != n) {
    Py_DECREF(result);
    return -1;
  }
  bool valid = true;
  for (int a = 0; a < n; ++a) {
    auto value = PyList_GetItem(result, a);
    if (PyBool_Check(value) || (!PyFloat_Check(value) && !PyLong_Check(value))) {
      valid = false;
      break;
    }
    weights[a] = PyFloat_AsDouble(value);
    if (PyErr_Occurred() || !std::isfinite(weights[a])) {
      PyErr_Clear();
      valid = false;
      break;
    }
  }
  Py_DECREF(result);
  return valid ? 1 : -1;
}
extern "C" PortfolioCallback sa_python_load_portfolio(const char *source, const char *assets,
                                                      const char *params) {
  return load_python(source, assets, params, "load_portfolio") ? python_on_portfolio : nullptr;
}
extern "C" void sa_python_close() {
  if (Py_IsInitialized()) {
    Py_XDECREF(callback);
    Py_XDECREF(globals);
    callback = nullptr;
    globals = nullptr;
    Py_FinalizeEx();
  }
}
