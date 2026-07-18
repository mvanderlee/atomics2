"""End-to-end tests for the public atomics API against the bundled patomic lib."""
import sys
from threading import Thread

import pytest

import atomics
from atomics import Alignment, CmpxchgResult, MemoryOrder, OpType
from atomics.exc import MemoryOrderError

WIDTHS = [1, 2, 4, 8]


# region - Integral operations

@pytest.mark.parametrize("width", WIDTHS, ids=[f"width{w}" for w in WIDTHS])
@pytest.mark.parametrize("atype", [
    pytest.param(atomics.INT, id="int"),
    pytest.param(atomics.UINT, id="uint"),
])
def test_arithmetic_ops(width, atype) -> None:
    a = atomics.atomic(width=width, atype=atype)
    assert a.load() == 0
    a.store(5)
    a.add(10)
    a.sub(3)
    a.inc()
    a.dec()
    assert a.load() == 12
    assert a.fetch_add(8) == 12
    assert a.fetch_sub(2) == 20
    assert a.load() == 18


def test_signed_roundtrip() -> None:
    a = atomics.atomic(width=4, atype=atomics.INT)
    a.store(-7)
    assert a.load() == -7
    a.neg()
    assert a.load() == 7
    assert a.signed


def test_uint_wraparound() -> None:
    u = atomics.atomic(width=4, atype=atomics.UINT)
    u.dec()
    assert u.load() == 0xFFFFFFFF
    assert not u.signed


def test_exchange() -> None:
    a = atomics.atomic(width=4, atype=atomics.INT)
    a.store(3)
    assert a.exchange(9) == 3
    assert a.load() == 9

# endregion - Integral operations


# region - Compare-exchange

def test_cmpxchg_loop() -> None:
    a = atomics.atomic(width=4, atype=atomics.INT)
    a.store(6)
    res = CmpxchgResult(success=False, expected=a.load())
    while not res:
        res = a.cmpxchg_weak(expected=res.expected, desired=res.expected * 7)
    assert a.load() == 42


def test_cmpxchg_failure_reports_actual_value() -> None:
    a = atomics.atomic(width=4, atype=atomics.INT)
    a.store(42)
    res = a.cmpxchg_strong(expected=1, desired=2)
    assert not res.success
    assert res.expected == 42
    assert a.load() == 42


def test_failed_cmpxchg_does_not_mutate_caller_bytes() -> None:
    """Regression: patomic used to write the actual value through the caller's
    immutable ``expected`` bytes (corrupting interned width-1 singletons)."""
    b = atomics.atomic(width=1, atype=atomics.BYTES)
    b.store(b"\x07")
    expected = b"\x00"
    res = b.cmpxchg_strong(expected=expected, desired=b"\x01")
    assert not res.success
    assert res.expected == b"\x07"
    # index checks stay valid even if the interned singleton itself is corrupted
    assert expected[0] == 0
    assert (0).to_bytes(1, "little")[0] == 0
    assert bytes([0])[0] == 0
    i = atomics.atomic(width=1, atype=atomics.UINT)
    i.store(9)
    res = i.cmpxchg_strong(expected=3, desired=4)
    assert not res.success and res.expected == 9
    assert (3).to_bytes(1, "little")[0] == 3

# endregion - Compare-exchange


# region - Bytes, bitwise, binary operations

def test_bytes_store_load() -> None:
    b = atomics.atomic(width=2, atype=atomics.BYTES)
    b.store(b"\x0f\x00")
    assert b.load() == b"\x0f\x00"
    assert bytes(b) == b"\x0f\x00"


def test_bitwise_ops() -> None:
    b = atomics.atomic(width=2, atype=atomics.BYTES)
    # bit offsets index the native-endian integer representation
    b.store((1).to_bytes(2, sys.byteorder))
    assert b.bit_test(0) is True
    assert b.bit_test(1) is False
    assert b.bit_test_set(4) is False
    assert b.bit_test(4) is True
    assert b.bit_test_reset(4) is True
    assert b.bit_test_compl(0) is True
    assert b.bit_test(0) is False


def test_binary_ops() -> None:
    b = atomics.atomic(width=2, atype=atomics.BYTES)
    b.store(b"\x0f\x00")
    b.bin_or(b"\xf0\x00")
    assert b.load()[0] == 0xFF
    assert b.bin_fetch_and(b"\x0f\xff")[0] == 0xFF
    assert b.load()[0] == 0x0F
    b.bin_xor(b"\x0f\x00")
    assert b.load()[0] == 0x00

# endregion - Bytes, bitwise, binary operations


# region - Views and lifetime contract

def test_view_shared_buffer() -> None:
    buf = bytearray(4)
    with atomics.atomicview(buffer=buf, atype=atomics.INT) as v:
        v.store(1234)
        assert v.load() == 1234
    assert int.from_bytes(bytes(buf), sys.byteorder, signed=True) == 1234


def test_readonly_view_supports_load_only() -> None:
    ro = memoryview(bytes(4))
    with atomics.atomicview(buffer=ro, atype=atomics.INT) as v:
        assert v.load() == 0
        assert v.readonly
        assert OpType.LOAD in v.ops_supported
        assert OpType.STORE not in v.ops_supported


def test_view_context_contract() -> None:
    buf = bytearray(4)
    ctx = atomics.atomicview(buffer=buf, atype=atomics.INT)
    with ctx as v:
        with pytest.raises(ValueError):
            ctx.__enter__()
        with pytest.raises(ValueError):
            ctx.release()
        v.load()
    ctx.release()  # multiple release after exit is fine
    with pytest.raises(ValueError):
        ctx.__enter__()

# endregion - Views and lifetime contract


# region - Alignment, properties, memory orders

def test_alignment() -> None:
    al = Alignment(4)
    assert al.recommended >= 1
    assert al.is_valid(bytearray(4))
    assert al.is_valid_recommended(bytearray(4))


def test_unsupported_width_raises() -> None:
    with pytest.raises(atomics.exc.UnsupportedWidthException):
        Alignment(3000)


def test_properties() -> None:
    a = atomics.atomic(width=4, atype=atomics.INT)
    assert a.width == 4
    assert not a.readonly
    assert len(a.ops_supported) > 10


@pytest.mark.parametrize("order,op", [
    pytest.param(MemoryOrder.RELEASE, "load", id="release-load"),
    pytest.param(MemoryOrder.ACQUIRE, "store", id="acquire-store"),
])
def test_invalid_memory_order_raises(order, op) -> None:
    a = atomics.atomic(width=4, atype=atomics.INT)
    with pytest.raises(MemoryOrderError):
        getattr(a, op)(*([1] if op == "store" else []), order=order)


def test_valid_memory_orders() -> None:
    a = atomics.atomic(width=4, atype=atomics.INT)
    a.store(1, order=MemoryOrder.RELEASE)
    assert a.load(order=MemoryOrder.ACQUIRE) == 1

# endregion - Alignment, properties, memory orders


# region - Concurrency

def test_threaded_increment() -> None:
    a = atomics.atomic(width=4, atype=atomics.INT)
    n = 50_000
    threads = [Thread(target=lambda: [a.inc() for _ in range(n)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert a.load() == 4 * n

# endregion - Concurrency
