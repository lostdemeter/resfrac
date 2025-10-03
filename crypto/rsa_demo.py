# resfrac/crypto/rsa_demo.py
# Simple demo: generate an RSA keypair and encrypt/decrypt a small UTF-8 message.
# Usage examples (run from project root /home/thorin/resfrac):
#   python -m resfrac.crypto.rsa_demo --bits 1024 --msg "hello resfrac"
#   python -m resfrac.crypto.rsa_demo --bits 1024 --msg "hello" --use-srt

import argparse
import sys

from resfrac.crypto.rsa import generate_rsa_keypair


def _encode_message_to_int(msg: str) -> int:
    data = msg.encode("utf-8")
    return int.from_bytes(data, byteorder="big", signed=False)


def _decode_int_to_message(x: int) -> str:
    if x == 0:
        return ""
    blen = (x.bit_length() + 7) // 8
    data = x.to_bytes(blen, byteorder="big", signed=False)
    try:
        return data.decode("utf-8", errors="strict")
    except Exception:
        # Fallback for any non-UTF8 content
        return data.decode("utf-8", errors="replace")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="RSA demo: generate keys and encrypt/decrypt a small message")
    p.add_argument("--bits", type=int, default=1024, help="RSA modulus size in bits (>=1024)")
    p.add_argument("--msg", type=str, default="hello resfrac", help="Small UTF-8 message to encrypt")
    p.add_argument("--use-srt", action="store_true", help="Use SRT for primality checks when available")
    args = p.parse_args(argv)

    # Choose primality function
    is_prime_fn = None
    if args.use_srt:
        try:
            from resfrac.crypto.rsa_srt_adapter import srt_is_prime
            is_prime_fn = srt_is_prime
        except Exception as e:
            print(f"Warning: --use-srt requested but unavailable ({e}). Falling back to internal MR.")
            is_prime_fn = None

    # Generate keypair
    if is_prime_fn is None:
        pub, priv = generate_rsa_keypair(bits=args.bits)
    else:
        pub, priv = generate_rsa_keypair(bits=args.bits, is_prime_fn=is_prime_fn)

    n, e = pub
    _, d, *_ = priv

    # Encode message to integer
    m = _encode_message_to_int(args.msg)
    if m >= n:
        print(
            "Error: Message integer is >= modulus n. Use a shorter message or larger --bits.",
            file=sys.stderr,
        )
        print(f"message_bits={m.bit_length()} n_bits={n.bit_length()}", file=sys.stderr)
        return 2

    # Encrypt/decrypt
    c = pow(m, e, n)
    m2 = pow(c, d, n)
    msg_out = _decode_int_to_message(m2)

    ok = (m == m2)

    print(f"bits: {args.bits}")
    print(f"n bits: {n.bit_length()} e: {e}")
    print(f"message: {args.msg}")
    print(f"cipher (hex): {hex(c)}")
    print(f"decrypted ok: {ok}")
    print(f"decrypted message: {msg_out}")

    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
