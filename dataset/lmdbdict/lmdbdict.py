import lmdb
import pickle
import os
from .utils import PicklableWrapper, picklable_wrapper
from .methods import DUMPS_FUNC, LOADS_FUNC

RESERVED = [
    b'__keys__',
    b'__key_dumps__',
    b'__value_dumps__',
    b'__key_loads__',
    b'__value_loads__',
]

class lmdbdict:
    def __init__(self, lmdb_path, mode='r',
                 key_method=None, value_method=None,
                 key_dumps=None, key_loads=None,
                 value_dumps=None, value_loads=None,
                 unsafe=False,subdir=False,map_size=1_099_511_627_776 * 2,max_readers=100,
                 readahead=False):
        """
        Args:
        value/key_dumps/loads: can be picklable functions
        or str or None
        if None: then default pickle
        if 'identity' then func = lambda x: x
        if saved in the db, then use what's in db
        unsafe: if True, you can getitem by the key even the key is not
        in the self._keys.
        subdir: if True, write data and lock files in a dir
        readahead: for lmdb reader, only make sense when mode='r'
        """
        self.map_size = map_size
        self.max_readers = max_readers
        self.lmdb_path = lmdb_path
        self.mode = mode
        self.readahead = readahead
        self.subdir = subdir
        self._init_db()
        if self.db_txn.get(b'__keys__'):
            try:
                self._keys = pickle.loads(self.db_txn.get(b'__keys__'))
            except:
                print('cant decode the keys saved in the lmdb, leave it empty now')
                self._keys = set()
                if self.mode == 'w':
                    print('Warning: any change you make under write mode may not be revertable.')
        else:  # no keys
            self._keys = []
            if self.mode == 'r':
                print('Reading an empty lmdb')
        # This is for backward compatiblility, starting from 0.3
        # _keys should always be a list
        if type(self._keys) is set:
            self._keys = sorted(list(self._keys), key=lambda x:pickle.dumps(x))

        self._init_dumps_loads(value_method, value_dumps, value_loads, which='value')
        self._init_dumps_loads(key_method, key_dumps, key_loads, which='key')

        self.unsafe = unsafe


    def _init_dumps_loads(self, method, dumps, loads, which='value'):
        """
        Initialize the key/value dumps loads function according to
        the user input or the db.
        """

        if method is not None:
            assert dumps is None and loads is None, f'{which}_method and {which}_dumps/loads cannot both be non-None'
            dumps = loads = method

        # Since the dumps or loads may be saved into db
        # Make them picklable first
        # Note: If dumps and loads are already picklable like str or None
        # This functionwon't change them
        if callable(dumps) and callable(loads):
            dumps = PicklableWrapper(dumps)
            loads = PicklableWrapper(loads)

        # The keys in the db
        db_dumps = f'__{which}_dumps__'.encode('ascii')
        db_loads = f'__{which}_loads__'.encode('ascii')

        if self.db_txn.get(db_dumps) is not None and\
           self.db_txn.get(db_loads) is not None:
            saved_dumps = pickle.loads(self.db_txn.get(db_dumps))
            saved_loads = pickle.loads(self.db_txn.get(db_loads))
            assert dumps is None and loads is None, \
                f'{which}_dumps/loads/method have to be None when read from a non-empty lmdb'
            # assert (getattr(dumps, '_obj', dumps) == saved_dumps or dumps is None) \
            #     and (getattr(loads, '_obj', loads) == saved_loads or loads is None), \
            #     f'{which}_dumps and {which}_loads has to be the same as what\'s saved in the lmdb. Or just feed None'
            dumps, loads = saved_dumps, saved_loads
            # Make them multiprocessing forkable
            if callable(dumps) and callable(loads):
                dumps = PicklableWrapper(dumps)
                loads = PicklableWrapper(loads)
        elif self.mode == 'w':
            # Write to the db_txn
            self.db_txn.put(db_dumps, pickle.dumps(dumps))
            self.db_txn.put(db_loads, pickle.dumps(loads))
            self.db_txn.commit()
            self.db_txn = self.env.begin(write=True)
        elif self.mode == 'r':
            # Note, here there is no value dumps and loads in db
            # Will use default pickle
            assert dumps is None and loads is None, \
                f'cannot set the {which}_dumps and {which}_loads under read mode'
            print(f"No {which} dumps and loads found in lmdb, will use pickle")

        if dumps is None or loads is None:
            assert dumps == loads, f'The {which}_dumps and {which}_loads have to be both None'
            setattr(self, f'_{which}_dumps', pickle.dumps)
            setattr(self, f'_{which}_loads', pickle.loads)
        elif type(dumps) is str and type(loads) is str:
            assert dumps == loads, f'The {which}_dumps and {which}_loads have to correspondant'
            setattr(self, f'_{which}_dumps', DUMPS_FUNC[dumps])
            setattr(self, f'_{which}_loads', LOADS_FUNC[loads])
        else:  # have to be function
            setattr(self, f'_{which}_dumps', dumps)
            setattr(self, f'_{which}_loads', loads)

    def keys(self):
        return self._keys

    def __contains__(self, item):
        return item in self._keys

    def __getstate__(self):
        r"""
        Make it pickable
        """
        state = self.__dict__
        state["env"] = None
        state["db_txn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._init_db()

    def _init_db(self):
        if self.mode == 'r':
            self.env = lmdb.open(
                self.lmdb_path,
                subdir=os.path.isdir(self.lmdb_path),
                readonly=self.readahead, lock=False,
                # readonly=True, lock=True,
                readahead=False, map_size=self.map_size,
                max_readers=self.max_readers,
            )
            self.db_txn = self.env.begin(write=False)

        elif self.mode == 'w':
            self.env = lmdb.open(
                self.lmdb_path, subdir=self.subdir,
                readonly=False, map_size=1099511627776 * 2,
                meminit=False, map_async=True)
            self.db_txn = self.env.begin(write=True)

    def __getitem__(self, key):
        if not self.unsafe:
            # Under safe mode, the key has to be in the self._keys
            if not key in self:
                raise KeyError
        tmp = self.db_txn.get(self._key_dumps(key,protocol=4)) or self.db_txn.get(self._key_dumps(key,protocol=3)) 

        if tmp is None:
            raise KeyError
        else:
            return self._value_loads(tmp)

    def __setitem__(self, key, value):
        assert self.mode == 'w', 'can only write item in write mode'
        # in fact even key is __len__ it should be fine, because it's dumped in pickle mode.
        assert key not in ['__keys__'], \
            f'{key} is internal variable, immutable to users'
        self.db_txn.put(self._key_dumps(key), self._value_dumps(value))
        self._keys.append(key)  # only update to the lmdb after flush

    def __delitem__(self, key):
        assert self.mode == 'w', 'can only write item in write mode'
        assert key in self._keys, f'{key} not in this lmdb'

        self.db_txn.delete(self._key_dumps(key)) or self.db_txn.delete(self._key_dumps(key, protocol=3))
        self._keys.remove(key)

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError

    def update(self, d):
        assert self.mode == 'w'
        for k, v in d.items():
            self[k] = v

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.lmdb_path + ')'

    def __del__(self):
        # make sure the the __key__ and __len__ are updated
        # and it's flushed
        if self.mode == 'w':
            self.flush()
            self.env.sync()
            self.env.close()

    def flush(self):
        assert self.mode == 'w', 'only flush when in write mode'
        # update __keys__ value
        self.db_txn.put(b'__keys__', pickle.dumps(self._keys))
        self.db_txn.commit()
        self.db_txn = self.env.begin(write=True)

    def sequential_iter(self):
        c = self.db_txn.cursor()
        for k, v in c:
            if k not in RESERVED:
                yield (self._key_loads(k), self._value_loads(v))


# TODO separate the logic between lmdb handling and key, value dumps.

# Aliasing
LMDBDict = lmdbdict
