"""TensorRT engine building (with an on-disk cache) and a minimal executor.

Engines are cached in engines/ under a key made from the ONNX bytes, the
TensorRT version, the GPU name, the precision and the batch profile, so a
TensorRT upgrade or a changed model rebuilds automatically.
"""
import hashlib
import json
import os

import cupy as cp
import numpy as np
import tensorrt as trt

_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engines")


def make_batch_dynamic(onnx_bytes, dim_name="N"):
    """Return ONNX bytes with dim 0 of every graph input/output made symbolic."""
    import onnx
    m = onnx.load_from_string(onnx_bytes)
    for v in list(m.graph.input) + list(m.graph.output):
        v.type.tensor_type.shape.dim[0].dim_param = dim_name
    return m.SerializeToString()


def _gpu_name():
    return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()


def engine_path(onnx_path, fp16=True, max_batch=1):
    data = open(onnx_path, "rb").read()
    key = hashlib.sha256(data + json.dumps([trt.__version__, _gpu_name(), fp16, max_batch]).encode())
    base = os.path.splitext(os.path.basename(onnx_path))[0]
    return os.path.join(ENGINE_DIR, f"{base}-b{max_batch}-{'fp16' if fp16 else 'fp32'}-{key.hexdigest()[:12]}.engine")


def build_engine(onnx_path, fp16=True, max_batch=1, workspace_mb=1024, force=False):
    """Build (or load from cache) an engine; returns its path."""
    path = engine_path(onnx_path, fp16, max_batch)
    if os.path.exists(path) and not force:
        return path
    os.makedirs(ENGINE_DIR, exist_ok=True)
    data = open(onnx_path, "rb").read()
    if max_batch > 1:
        data = make_batch_dynamic(data)
    builder = trt.Builder(_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, _LOGGER)
    if not parser.parse(data):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if max_batch > 1:
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = list(inp.shape)
            lo, opt, hi = [1] + shape[1:], [max_batch] + shape[1:], [max_batch] + shape[1:]
            profile.set_shape(inp.name, lo, opt, hi)
        config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    with open(path + ".tmp", "wb") as f:
        f.write(blob)
    os.replace(path + ".tmp", path)
    with open(path + ".json", "w") as f:
        json.dump({"onnx": os.path.abspath(onnx_path), "tensorrt": trt.__version__, "gpu": _gpu_name(),
                   "fp16": fp16, "max_batch": max_batch}, f, indent=1)
    return path


class _OutputAllocator(trt.IOutputAllocator):
    """For data-dependent output shapes (e.g. NMS results): hands TensorRT a
    preallocated buffer and records the actual shape it reports."""

    def __init__(self, nbytes):
        super().__init__()
        self.buf = cp.zeros(nbytes, cp.uint8)
        self.shape = None

    def reallocate_output(self, tensor_name, memory, size, alignment):
        if size > self.buf.nbytes:
            self.buf = cp.zeros(size, cp.uint8)
        return self.buf.data.ptr

    def reallocate_output_async(self, tensor_name, memory, size, alignment, stream):
        return self.reallocate_output(tensor_name, memory, size, alignment)

    def notify_shape(self, tensor_name, shape):
        self.shape = tuple(shape)


class TrtModel:
    """Runs an engine on a CUDA stream with preallocated CuPy I/O buffers.

    Inputs are written into model.inputs[name][:batch] (CuPy arrays), then
    run(batch) enqueues inference; outputs are model.outputs[name][:batch].
    Outputs whose size depends on the data (a -1 beyond the batch dim) are read
    with dynamic_output(name) after the stream is synchronized.
    Nothing is synchronized here; the caller syncs the stream when it needs results.
    """

    def __init__(self, engine_file, stream=None):
        self.runtime = trt.Runtime(_LOGGER)
        with open(engine_file, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = stream or cp.cuda.Stream(non_blocking=True)
        self.inputs, self.outputs, self.dynamic = {}, {}, {}
        self.max_batch = 1
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            if shape[0] == -1:
                hi = self.engine.get_tensor_profile_shape(name, 0)[2][0] \
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else None
                if hi:
                    self.max_batch = hi
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            raw_shape = list(self.engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT and -1 in raw_shape[1:]:
                alloc = _OutputAllocator(1 << 20)
                self.context.set_output_allocator(name, alloc)
                self.dynamic[name] = (alloc, dtype)
                continue
            shape = [self.max_batch if d == -1 else d for d in raw_shape]
            buf = cp.zeros(shape, dtype)
            self.context.set_tensor_address(name, buf.data.ptr)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = buf
            else:
                self.outputs[name] = buf
        self._batch = None

    def run(self, batch=1):
        if batch != self._batch:
            for name, buf in self.inputs.items():
                self.context.set_input_shape(name, (batch,) + buf.shape[1:])
            self._batch = batch
        if not self.context.execute_async_v3(self.stream.ptr):
            raise RuntimeError("TensorRT execute failed")

    def dynamic_output(self, name):
        """CuPy view of a data-dependent output (call after stream sync)."""
        alloc, dtype = self.dynamic[name]
        n = int(np.prod(alloc.shape)) if alloc.shape else 0
        return alloc.buf[:n * dtype.itemsize].view(dtype).reshape(alloc.shape)
