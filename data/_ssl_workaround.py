"""Opt-in TLS verification bypass for Hugging Face dataset downloads.

Shannon's campus network runs Fortinet SSL inspection on some Hugging Face
CDN hosts (seen on the "Xet" storage backend, e.g. us.aws.cdn.hf.co) and
re-signs them with its own CA. Neither the host OS nor any container on it
trusts that CA -- confirmed with `curl` directly on the host, which fails
identically -- and there's no way to add trust from inside a container here
(Shannon's Docker authorization policy rejects bind-mounting host system
paths). The real fix is getting the actual CA cert from Shannon's network
admin; until then this is a controlled, explicit trade-off.

Enabled via HF_INSECURE_SSL=1. Scoped to disabling verification only for the
`requests` calls huggingface_hub/datasets make while fetching these fully
public, unauthenticated datasets -- never used for anything carrying
secrets or tokens, and off by default everywhere else.
"""
import os


def apply_if_requested():
    if os.environ.get("HF_INSECURE_SSL") != "1":
        return
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    orig_request = requests.Session.request

    def unsafe_request(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        return orig_request(self, *args, **kwargs)

    requests.Session.request = unsafe_request
    print("[ssl-workaround] TLS verification disabled for HTTP dataset downloads "
          "(HF_INSECURE_SSL=1) -- see data/_ssl_workaround.py for why.")
