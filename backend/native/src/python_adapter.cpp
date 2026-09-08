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
extern "C" Callback sa_python_load(const char *source, const char *bars, const char *params) {
  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  config.use_environment = 0;
  config.write_bytecode = 0;
  PyConfig_SetBytesString(&config, &config.program_name, SA_PYTHON_EXECUTABLE);
  auto status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status))
    return nullptr;
  std::ifstream file(std::string(SA_BACKEND_ROOT) + "/simulation/python_strategy.py");
  std::string code{std::istreambuf_iterator<char>(file), {}};
  if (code.empty())
    return nullptr;
  globals = PyDict_New();
  PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins());
  PyObject *result = PyRun_String(code.c_str(), Py_file_input, globals, globals);
  if (!result) {
    PyErr_Print();
    return nullptr;
  }
  Py_DECREF(result);
  callback =
      PyObject_CallFunction(PyDict_GetItemString(globals, "load"), "sss", source, bars, params);
  if (!callback) {
    PyErr_Print();
    return nullptr;
  }
  return python_on_bar;
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
