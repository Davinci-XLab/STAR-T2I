# A simple dumps and loads function factory
import pickle

try:
    import pyarrow as pa
except ImportError:
    PYARROW_AVAILABLE = False
else:
    PYARROW_AVAILABLE = True


# Only use when you are sure the input is a byte
def identity(x):
    return x


def ascii_encode(x):
    return x.encode('ascii')


def ascii_decode(x):
    return x.decode('ascii')


def utf8_encode(x):
    return x.encode('utf8')


def utf8_decode(x):
    return x.decode('utf8')


def pa_dumps(x):
    assert PYARROW_AVAILABLE, 'pyarrow not installed'
    return pa.serialize(x).to_buffer()


def pa_loads(x):
    assert PYARROW_AVAILABLE, 'pyarrow not installed'
    return pa.deserialize(x)


DUMPS_FUNC = dict(
    identity=identity,
    ascii=ascii_encode,
    utf8=utf8_encode,
    pyarrow=pa_dumps,
    pickle=pickle.dumps,
)

LOADS_FUNC = dict(
    identity=identity,
    ascii=ascii_decode,
    utf8=utf8_decode,
    pyarrow=pa_loads,
    pickle=pickle.loads,
)