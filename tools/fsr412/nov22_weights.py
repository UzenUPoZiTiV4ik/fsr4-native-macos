# SPDX-License-Identifier: MIT
"""Static import of the normal <=4K Lite Lossy network from the Nov 22 RX 6000 DLL."""
import struct
from pathlib import Path

from fsr412 import lossy_weights as dxil

RELEASE = 'dll-2026-11-22'
URL = ('https://github.com/zgauthier2000/fsr4-shader-performance-overrides/releases/download/'
       + RELEASE + '/fsr4.1.2-cyboman-rx6000-2026-11-22.zip')
ARCHIVE_SHA256 = '2d81d93647e5c78c04401cf1437bd706e2c634ac72ed051817571c5c98be7d15'
DLL_SHA256 = '5f17b595b5abd9fbe19178f21912a4046bbe560929383ef889a43f7109110a43'
LOSSY_INITIALIZER_SHA256 = '899dc09a238930f67742165d0c8f741150cc3ec83780198588256aaec655d981'
BIAS_MATRIX_SHA256 = 'ddcffcb13fc47521911dd3c10d2f309c3ef4da18e3b55425e3cf2af96ae68250'
BUFFER_SHA256 = '4a6edad16b05f87f54b192344b1c7fec39b82a9ed10bc4cd18f3d2290e6f8fa4'
BUFFER_OFFSETS = (0xc04a00, 0xc44a20, 0xc64a30, 0xc84a40, 0xca4a40)
POST_CONTAINERS = {
    1080: 'd6fdbe53e15ee4a8830abac979f569958932d5fe39d1a5ce687b22009ac0adbe',
    2160: '74c94eb91891038073b333ad1650819e365ee425e69143b5cd4823ecfdd7a28e',
}


def recover(dll, dxc, base_initializer):
    image = Path(dll).read_bytes()
    base = Path(base_initializer).read_bytes()
    if dxil.digest(image) != DLL_SHA256 or dxil.digest(base) != dxil.COMMON_SHA256:
        raise ValueError('unexpected Lite Lossy DLL or AMD base initializer')
    buffers = [image[offset:offset + dxil.INITIALIZER_BYTES] for offset in BUFFER_OFFSETS]
    if any(dxil.digest(buffer) != BUFFER_SHA256 for buffer in buffers):
        raise ValueError('trained ratio buffers disagree with the release pin')

    # The first three passes use these trained tables. POST additionally bakes
    # biases into the four CNB clones and the subpixel-specific CT tails.
    start, size = dxil._section(image)
    decoded = {}
    for _, container in dxil.dxbc_containers(image, start, start + size):
        identity = dxil.digest(container)
        if identity not in POST_CONTAINERS.values():
            continue
        listing = dxil.disassemble(dxc, container)
        tier = dxil._tier(listing, 'post')
        if POST_CONTAINERS.get(tier) != identity:
            raise ValueError('unexpected POST tier')
        # Twelve extra signed-dot instructions per CNB clone implement the
        # residual epilogue. Keep only the convolution calls for table recovery.
        epilogue = [line for line in listing.splitlines() if line.lstrip().startswith('%mt2.') and 'dot4AddPacked' in line]
        if len(epilogue) != 48:
            raise ValueError('unexpected POST residual layout')
        exclude = set(epilogue)
        weights_listing = '\n'.join(line for line in listing.splitlines() if line not in exclude)
        decoded[tier] = dxil.decode_post(weights_listing, base)
    if set(decoded) != set(POST_CONTAINERS) or decoded[1080] != decoded[2160]:
        raise ValueError('trained POST tables differ or a tier is missing')
    tables, conditional = decoded[1080]
    data = bytearray(buffers[0])
    for name, offset, length in dxil.LAYOUT['post']:
        if len(tables[name]) != length:
            raise ValueError('invalid POST table size')
        data[offset:offset + length] = tables[name]
    matrix = [list(struct.unpack('<8i', tables['bCT'])) for _ in range(4)]
    for change in conditional:
        if change['layer'] == 'CT':
            matrix[change['tap']][change['channel']] = change['conditional_bias']
    return bytes(data), {
        'schema': 'fsr412_lite_lossy_import_v1',
        'source_dll_sha256': DLL_SHA256,
        'initializers': {'initializers-lite-lossy.bin': dxil.digest(data)},
        'ratio_buffer_offsets': list(BUFFER_OFFSETS),
        'ratio_buffer_sha256': BUFFER_SHA256,
        'post_containers': POST_CONTAINERS,
        'conditional_biases': conditional,
        'post_ct_biases_by_tap': matrix,
        'supported_trained_tiers': [1080, 2160],
    }
