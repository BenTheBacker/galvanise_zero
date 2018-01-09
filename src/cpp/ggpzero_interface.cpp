// python includes
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include "rng.h"

///////////////////////////////////////////////////////////////////////////////
// global variable

PyObject* ggpzero_interface_error;

#include "pyobjects/py_bases.cpp"
#include "pyobjects/py_supervisor.cpp"

///////////////////////////////////////////////////////////////////////////////

static PyObject* GGPZero_Interface_hello(PyObject* self, PyObject* args) {
    // XXX what is self in this context?

    const char* name = nullptr;
    if (! ::PyArg_ParseTuple(args, "s", &name)) {
        return nullptr;
    }

    std::string msg = K273::fmtString("Hello world %s", name);

    xoroshiro32plus16 random;
    for (int ii=0; ii<10000; ii++) {
        K273::l_verbose("random/42 %d", random.getWithMax(42));
        K273::l_verbose("random/real %.4f", (random() / (double) random.max()));
    }

    return ::Py_BuildValue("s", msg.c_str());
}

///////////////////////////////////////////////////////////////////////////////

PyMethodDef gi_functions[] = {
    {"hello_test", GGPZero_Interface_hello, METH_VARARGS, "hello_test"},
    {"GdlBasesTransformer", gi_GdlBasesTransformer, METH_VARARGS, "GdlBasesTransformer"},
    {"Supervisor", gi_Supervisor, METH_VARARGS, "Supervisor"},
    {nullptr, nullptr, 0, nullptr}
};

///////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

    PyMODINIT_FUNC initggpzero_interface() {
        Py_TYPE(&PyType_GdlBasesTransformerWrapper) = &PyType_Type;

        if (::PyType_Ready(&PyType_GdlBasesTransformerWrapper) < 0) {
            return;
        }

        PyObject* m = ::Py_InitModule("ggpzero_interface", gi_functions);
        if (m == nullptr) {
            return;
        }

        char error_name[] = "ggpzero_interface.error";
        ggpzero_interface_error = ::PyErr_NewException(error_name, nullptr, nullptr);
        Py_INCREF(ggpzero_interface_error);
        ::PyModule_AddObject(m, "AbcModuleError", ggpzero_interface_error);

        import_array();
    }
}


