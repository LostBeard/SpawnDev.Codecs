# NOTICE

SpawnDev.Codecs is licensed under the MIT License (see `LICENSE.txt`).

This project incorporates, consults, or is structurally inspired by the following open-source works. Their licenses and copyright notices are reproduced below and acknowledged here.

---

## Opus reference implementation - libopus (planned for Phase 1)

**Copyright:** © 2001-2011 Xiph.Org, Skype Limited, Octasic, Jean-Marc Valin, Timothy B. Terriberry, CSIRO, Gregory Maxwell, Mark Borgerding, Erik de Castro Lopo
**License:** BSD 3-Clause (Xiph.Org license)
**Source:** https://github.com/xiph/opus

When SpawnDev.Codecs consults or ports structure from libopus (e.g. the SILK layer, range coder, or CELT reference algorithms), this notice and the upstream license apply to the ported portions.

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## Concentus - pure-C# libopus port (runtime dependency for the CELT path)

**Copyright:** © 2016 Logan Stromberg (and libopus original copyright holders)
**License:** BSD 3-Clause (inherited from libopus)
**Source:** https://github.com/lostromb/concentus

Concentus is a BSD-3-Clause pure-C# port of libopus 1.1.2.

**Status as of CELT decoder + Opus encoder slices:** SpawnDev.Codecs depends
on Concentus at runtime through
`<PackageReference Include="Concentus" Version="2.2.2" />` in
`SpawnDev.Codecs.csproj`. The CELT decode path in
`Audio/Opus/Celt/CeltDecoder.cs` and the top-level
`Audio/Opus/OpusEncoder.cs` both delegate the heavy lifting to the Concentus
implementation:

* **CELT decode:** range coding, bit allocation, energy decode, PVQ shape
  decode, IMDCT, comb-filter, de-emphasis.
* **Opus encode:** SILK / CELT / Hybrid mode selection, rate control, full
  per-frame analysis + entropy coding, packet framing.

This delivers bit-exact-vs-libopus CELT decode and a fully working RFC 6716
Opus encoder today, and gives a verifiable oracle to compare future
hand-ports against.

The local files in `Audio/Opus/Celt/` (`CeltConstants`, `CeltMode`,
`CeltDecoderState`) and `Audio/Opus/Silk/` scaffold the eventual hand-ports:
when the per-module hand-ports replace the Concentus delegation inside
`CeltDecoder` and `OpusEncoder`, the public API surface and the
`OpusDecoder` / `OpusEncoder` integration do not change. BSD-3 terms
propagate to all consuming distributions of SpawnDev.Codecs.

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the names of
  specific contributors, may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## RFC 6716 (Opus codec specification)

**Authors:** Jean-Marc Valin (Mozilla), Koen Vos (Skype), Timothy B. Terriberry (Mozilla)
**Publisher:** IETF (Internet Engineering Task Force)
**URL:** https://datatracker.ietf.org/doc/rfc6716/

SpawnDev.Codecs.Opus is an independent implementation of the Opus codec as defined by RFC 6716 and its updates (RFC 8251).

---

## RFC 6386 (VP8 codec specification) - planned for Phase 2

**Authors:** Jim Bankoski, Paul Wilkins, Yaowu Xu
**Publisher:** IETF
**URL:** https://datatracker.ietf.org/doc/rfc6386/

---

## libvpx - reference VP8/VP9 implementation (planned for Phase 2+)

**Copyright:** © 2010-present, The WebM Project Authors, Google LLC
**License:** BSD 3-Clause (VP8/VP9 license grant)
**Source:** https://github.com/webmproject/libvpx

VP8 and VP9 are released under the Google WebM patent pledge - patent-clean for royalty-free implementation.

---

## libaom - reference AV1 implementation (planned for later phase)

**Copyright:** © 2018-present, Alliance for Open Media
**License:** BSD 2-Clause + patent grant
**Source:** https://aomedia.googlesource.com/aom/

AV1 is released under the AOMedia patent license - patent-clean for royalty-free implementation.

---

## libFLAC / libvorbis - reference FLAC and Vorbis implementations (planned for later phase)

**Copyright:** © Xiph.Org Foundation
**License:** BSD 3-Clause
**Source:** https://gitlab.xiph.org/xiph/flac - https://gitlab.xiph.org/xiph/vorbis

---

## Final notes

All codecs SpawnDev.Codecs implements are **patent-clean** - Opus, VP8, VP9, AV1, FLAC, Vorbis. Patent-encumbered codecs (H.264, H.265, AAC) are out of scope for this library and are delegated to platform encoders via SpawnDev.MultiMedia.

For license questions or corrections, open an issue at https://github.com/LostBeard/SpawnDev.Codecs/issues.
