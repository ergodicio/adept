"""ctypes and DLPack boundary; importing this module does not initialize JAX."""

import ctypes as ct


class NativeCore:
    """Own one persistent MPI/native solver inside a worker process."""

    def __init__(self, library, options):
        self.lib = ct.CDLL(str(library))
        signatures = {
            "av_create": ([ct.c_int, ct.POINTER(ct.c_char_p)], ct.c_void_p),
            "av_error": ([], ct.c_char_p),
            "av_info": ([ct.c_void_p, ct.POINTER(ct.c_int)], ct.c_int),
            "av_step": ([ct.c_void_p, ct.c_int], ct.c_int),
            "av_tensor": ([ct.c_void_p, ct.c_int], ct.c_void_p),
            "av_set_field": ([ct.c_void_p, ct.c_char_p], ct.c_int),
            "av_broadcast": ([ct.c_char_p, ct.c_int], ct.c_int),
            "av_barrier": ([], ct.c_int),
            "av_stats": ([ct.c_void_p, ct.POINTER(ct.c_double)], ct.c_int),
            "av_close": ([ct.c_void_p], ct.c_int),
            "av_finalize": ([], ct.c_int),
            "av_abort": ([ct.c_int], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result
        argv = [b"adept-native", *(str(value).encode() for value in options)]
        self.handle = self.lib.av_create(len(argv), (ct.c_char_p * len(argv))(*argv))
        if not self.handle:
            raise RuntimeError(self.lib.av_error().decode())
        values = (ct.c_int * 7)()
        self.check(self.lib.av_info(self.handle, values))
        self.rank, self.size, self.local_rank, self.device_type, self.nx, self.nv, self.local_nv = values

    def check(self, result):
        if result:
            raise RuntimeError(self.lib.av_error().decode())

    def step(self, count):
        self.check(self.lib.av_step(self.handle, count))

    def broadcast(self, value):
        buffer = ct.create_string_buffer(value.encode(), 512)
        self.check(self.lib.av_broadcast(buffer, len(buffer)))
        return buffer.value.decode()

    def barrier(self):
        self.check(self.lib.av_barrier())

    def set_field(self, path):
        self.check(self.lib.av_set_field(self.handle, str(path).encode()))

    def view(self, field=False):
        return _View(self, field)

    def stats(self):
        values = (ct.c_double * 7)()
        self.check(self.lib.av_stats(self.handle, values))
        names = ["stream", "velocity", "pack", "mpi", "field", "collision", "total"]
        return dict(zip(names, values, strict=True))

    def close(self):
        if self.handle:
            self.check(self.lib.av_close(self.handle))
            self.handle = None
        self.check(self.lib.av_finalize())


class _View:
    """Synchronized, read-only-by-convention view valid until the next native step."""

    def __init__(self, owner, field):
        self.owner, self.field = owner, field

    def __dlpack_device__(self):
        return self.owner.device_type, self.owner.local_rank if self.owner.device_type == 2 else 0

    def __dlpack__(self, stream=None, *, max_version=None, dl_device=None, copy=None):
        if copy:
            raise BufferError("the native exporter provides views; the consumer may copy")
        if dl_device is not None and tuple(dl_device) != self.__dlpack_device__():
            raise BufferError("native DLPack export cannot change devices")
        pointer = self.owner.lib.av_tensor(self.owner.handle, int(self.field))
        if not pointer:
            raise RuntimeError(self.owner.lib.av_error().decode())
        # The consumer owns the tensor's C++ deleter after renaming this capsule
        # to used_dltensor. Its manager keeps the native allocation alive.
        capsule = ct.pythonapi.PyCapsule_New
        capsule.restype = ct.py_object
        capsule.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p]
        return capsule(pointer, b"dltensor", None)
