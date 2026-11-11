// photo-geolocator-01: a model-free photo geolocator.
//
// It places each of an instance's 25 street-level photos from its pixels alone.
// No model calls, no API key, no network but the env.
//
// How it works, per photo:
//   1. Decode the JPEG's DC coefficients (one average colour per 8x8 block), which
//      is a full low-resolution picture of the photo's colour layout, and turn it
//      into an 8x12 RGB grid.
//   2. Standardise that grid with the practice set's mean and spread, then compare
//      it to the 75 practice photos' grids (priors.json) with a Gaussian-weighted
//      k-nearest-neighbour vote over their known places.
//   3. Blend the vote with the practice set's geographic centre, by how close the
//      nearest neighbour is: a photo with no close colour match falls back to the
//      centre, so a wild guess never replaces a safe one.
//   4. Post every photo's latitude and longitude to the env, in photo order.
//
// Deterministic: every request is a function of the photo bytes and the fixed
// priors, the photos are fetched one at a time in order, and nothing reads a
// clock or an unseeded random source. A photo that fails to decode still gets
// the geographic centre, so all 25 are always answered.
//
// Runs as the grader runs an entry, `node /bot/main.ts` on Node 22, with
// MAKEFUN_ENV_URL (the env) and MAKEFUN_PROMPT_FILE (the brief, read only to
// confirm it is there). The priors live in priors.json beside this file.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

// ---------------------------------------------------------------- priors

interface Priors {
  count: number;
  rows: number;
  cols: number;
  featB64: string;
  latE6: number[];
  lonE6: number[];
  mean: number[];
  std: number[];
  priorLatE6: number;
  priorLonE6: number;
}

function loadPriors(): Priors {
  const candidates: string[] = ['/bot/priors.json'];
  const here = dirname(process.argv[1] ?? '');
  if (here) candidates.push(join(here, 'priors.json'));
  candidates.push('priors.json');
  for (const p of candidates) {
    try {
      return JSON.parse(readFileSync(p, 'utf8')) as Priors;
    } catch {
      // The next candidate.
    }
  }
  throw new Error('priors.json not found');
}

// ------------------------------------------------------- JPEG DC decoding
//
// A minimal baseline-JPEG reader: it walks the markers, reads the quantisation
// and Huffman tables and the frame, then decodes the scan just far enough to
// take every 8x8 block's DC coefficient, which is the block's average colour.
// AC coefficients are skipped, counting them, because a block whose 63 AC
// coefficients are all non-zero carries no end-of-block marker.

interface HuffTable {
  lut: Map<string, number>;
}

function buildHuff(counts: number[], symbols: number[]): HuffTable {
  const lut = new Map<string, number>();
  let code = 0;
  let k = 0;
  for (let len = 1; len <= 16; len += 1) {
    for (let n = 0; n < counts[len - 1]!; n += 1) {
      lut.set(`${len}:${code}`, symbols[k]!);
      k += 1;
      code += 1;
    }
    code <<= 1;
  }
  return { lut };
}

class BitReader {
  private readonly data: Uint8Array;
  private pos: number;
  private bitBuf = 0;
  private bitCnt = 0;

  constructor(data: Uint8Array, start: number) {
    this.data = data;
    this.pos = start;
  }

  private nextByte(): number {
    if (this.pos >= this.data.length) return -1;
    const b = this.data[this.pos]!;
    this.pos += 1;
    if (b === 0xff) {
      if (this.pos >= this.data.length) return -1;
      const nb = this.data[this.pos]!;
      if (nb === 0x00) {
        this.pos += 1; // Stuffed 0xFF00: a literal 0xFF in the data.
      } else if (nb === 0xd9) {
        return -1; // EOI marker.
      }
    }
    return b;
  }

  bits(n: number): number {
    while (this.bitCnt < n) {
      let b = this.nextByte();
      if (b < 0) b = 0;
      this.bitBuf = (this.bitBuf << 8) | b;
      this.bitCnt += 8;
    }
    this.bitCnt -= n;
    const v = (this.bitBuf >> this.bitCnt) & ((1 << n) - 1);
    this.bitBuf &= (1 << this.bitCnt) - 1;
    return v;
  }

  bit(): number {
    return this.bits(1);
  }
}

function huffDecode(h: HuffTable, br: BitReader): number {
  let code = 0;
  for (let len = 1; len <= 16; len += 1) {
    code = (code << 1) | br.bit();
    const s = h.lut.get(`${len}:${code}`);
    if (s !== undefined) return s;
  }
  throw new Error('bad Huffman code');
}

function recvExtend(br: BitReader, s: number): number {
  if (s === 0) return 0;
  const v = br.bits(s);
  if (v < 1 << (s - 1)) return v - ((1 << s) - 1);
  return v;
}

interface CompInfo {
  id: number;
  hs: number;
  vs: number;
  tq: number;
}

interface Frame {
  w: number;
  h: number;
  comps: CompInfo[];
}

/** Per-component grids of 8x8-block average samples (Y, Cb, Cr), level-shifted back. */
function decodeDcGrid(jpeg: Uint8Array): { frame: Frame; grids: Map<number, number[][]> } {
  if (jpeg[0] !== 0xff || jpeg[1] !== 0xd8) throw new Error('not a JPEG');
  let i = 2;
  const quant = new Map<number, number[]>();
  const huff = new Map<string, HuffTable>();
  let frame: Frame | null = null;
  let scanComps: { id: number; tdc: number; tac: number }[] | null = null;
  let scanStart = 0;
  const u16 = (at: number): number => (jpeg[at]! << 8) | jpeg[at + 1]!;
  while (i < jpeg.length) {
    if (jpeg[i] !== 0xff) throw new Error(`bad marker at ${i}`);
    const m = jpeg[i + 1]!;
    if (m === 0xda) {
      const len = u16(i + 2);
      const ns = jpeg[i + 4]!;
      scanComps = [];
      for (let c = 0; c < ns; c += 1) {
        const o = i + 5 + 2 * c;
        scanComps.push({ id: jpeg[o]!, tdc: jpeg[o + 1]! >> 4, tac: jpeg[o + 1]! & 15 });
      }
      scanStart = i + 2 + len;
      break;
    }
    const len = u16(i + 2);
    const segStart = i + 4;
    const segEnd = i + 2 + len;
    if (m === 0xdb) {
      // DQT
      let j = segStart;
      while (j < segEnd) {
        const pq = jpeg[j]! >> 4;
        const tq = jpeg[j]! & 15;
        j += 1;
        const vals: number[] = [];
        for (let k = 0; k < 64; k += 1) {
          if (pq === 0) {
            vals.push(jpeg[j]!);
            j += 1;
          } else {
            vals.push(u16(j));
            j += 2;
          }
        }
        quant.set(tq, vals);
      }
    } else if (m === 0xc4) {
      // DHT
      let j = segStart;
      while (j < segEnd) {
        const tc = jpeg[j]! >> 4;
        const th = jpeg[j]! & 15;
        j += 1;
        const counts: number[] = [];
        for (let k = 0; k < 16; k += 1) {
          counts.push(jpeg[j]!);
          j += 1;
        }
        const total = counts.reduce((a, b) => a + b, 0);
        const syms: number[] = [];
        for (let k = 0; k < total; k += 1) {
          syms.push(jpeg[j]!);
          j += 1;
        }
        huff.set(`${tc}:${th}`, buildHuff(counts, syms));
      }
    } else if (m === 0xc0 || m === 0xc1) {
      // SOF0/SOF1 (baseline)
      const h = u16(segStart + 1);
      const w = u16(segStart + 3);
      const nc = jpeg[segStart + 5]!;
      const comps: CompInfo[] = [];
      for (let c = 0; c < nc; c += 1) {
        const o = segStart + 6 + 3 * c;
        comps.push({ id: jpeg[o]!, hs: jpeg[o + 1]! >> 4, vs: jpeg[o + 1]! & 15, tq: jpeg[o + 2]! });
      }
      frame = { w, h, comps };
    }
    i += 2 + len;
  }
  if (frame === null || scanComps === null) throw new Error('no frame or scan');
  const byId = new Map<number, CompInfo>();
  for (const c of frame.comps) byId.set(c.id, c);
  const scan: (CompInfo & { tdc: number; tac: number })[] = [];
  for (const s of scanComps) {
    const c = byId.get(s.id);
    if (c === undefined) throw new Error('scan component not in frame');
    scan.push({ ...c, tdc: s.tdc, tac: s.tac });
  }
  const hmax = Math.max(...frame.comps.map((c) => c.hs));
  const vmax = Math.max(...frame.comps.map((c) => c.vs));
  const mcux = Math.ceil(frame.w / (8 * hmax));
  const mcuy = Math.ceil(frame.h / (8 * vmax));
  const grids = new Map<number, number[][]>();
  for (const c of frame.comps) {
    const bw = Math.ceil((frame.w * c.hs) / (8 * hmax));
    const bh = Math.ceil((frame.h * c.vs) / (8 * vmax));
    const g: number[][] = [];
    for (let r = 0; r < bh; r += 1) g.push(new Array<number>(bw).fill(0));
    grids.set(c.id, g);
  }
  const br = new BitReader(jpeg, scanStart);
  const pred = new Map<number, number>();
  for (const c of frame.comps) pred.set(c.id, 0);
  for (let my = 0; my < mcuy; my += 1) {
    for (let mx = 0; mx < mcux; mx += 1) {
      for (const c of scan) {
        for (let vy = 0; vy < c.vs; vy += 1) {
          for (let hx = 0; hx < c.hs; hx += 1) {
            const dcTable = huff.get(`0:${c.tdc}`);
            const acTable = huff.get(`1:${c.tac}`);
            if (dcTable === undefined || acTable === undefined) throw new Error('missing Huffman table');
            const s = huffDecode(dcTable, br);
            const diff = recvExtend(br, s);
            const p = (pred.get(c.id) ?? 0) + diff;
            pred.set(c.id, p);
            const dc = p * (quant.get(c.tq)?.[0] ?? 1);
            const avg = dc / 8 + 128;
            const bx = mx * c.hs + hx;
            const by = my * c.vs + vy;
            const g = grids.get(c.id)!;
            if (by < g.length && bx < g[0]!.length) g[by]![bx] = avg;
            // Skip the AC coefficients, counting them: a block with all 63
            // AC coefficients non-zero has no end-of-block marker.
            let pos = 1;
            while (pos < 64) {
              const rs = huffDecode(acTable, br);
              if (rs === 0x00) break; // EOB
              const r = rs >> 4;
              const s2 = rs & 15;
              if (r === 15 && s2 === 0) {
                pos += 16; // ZRL
                continue;
              }
              if (s2 > 0) br.bits(s2);
              pos += r + 1;
            }
          }
        }
      }
    }
  }
  return { frame, grids };
}

/** The Y-resolution RGB grid (one entry per 8x8 Y block), from the DC grids. */
function rgbGrid(frame: Frame, grids: Map<number, number[][]>): number[][][] {
  const yId = frame.comps[0]!.id;
  const cbId = frame.comps[1]!.id;
  const crId = frame.comps[2]!.id;
  const Y = grids.get(yId)!;
  const Cb = grids.get(cbId)!;
  const Cr = grids.get(crId)!;
  const hs = frame.comps[0]!.hs;
  const vs = frame.comps[0]!.vs;
  const H = Y.length;
  const W = Y[0]!.length;
  const out: number[][][] = [];
  for (let by = 0; by < H; by += 1) {
    const row: number[][] = [];
    for (let bx = 0; bx < W; bx += 1) {
      const y = Y[by]![bx]!;
      const cb = Cb[Math.min(Math.floor(by / vs), Cb.length - 1)]![Math.min(Math.floor(bx / hs), Cb[0]!.length - 1)]! - 128;
      const cr = Cr[Math.min(Math.floor(by / vs), Cr.length - 1)]![Math.min(Math.floor(bx / hs), Cr[0]!.length - 1)]! - 128;
      const r = y + 1.402 * cr;
      const g = y - 0.34414 * cb - 0.71414 * cr;
      const b = y + 1.772 * cb;
      row.push([Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b))]);
    }
    out.push(row);
  }
  return out;
}

function downsample(rgb: number[][][], rows: number, cols: number): number[] {
  const H = rgb.length;
  const W = rgb[0]!.length;
  const feat: number[] = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const y0 = Math.floor((r * H) / rows);
      const y1 = Math.max(y0 + 1, Math.floor(((r + 1) * H) / rows));
      const x0 = Math.floor((c * W) / cols);
      const x1 = Math.max(x0 + 1, Math.floor(((c + 1) * W) / cols));
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let n = 0;
      for (let by = y0; by < y1; by += 1) {
        for (let bx = x0; bx < x1; bx += 1) {
          const px = rgb[by]![bx]!;
          sr += px[0]!;
          sg += px[1]!;
          sb += px[2]!;
          n += 1;
        }
      }
      feat.push(sr / n / 255, sg / n / 255, sb / n / 255);
    }
  }
  return feat;
}

// ------------------------------------------------------------------ guessing

function latLonToXyz(latDeg: number, lonDeg: number): [number, number, number] {
  const la = (latDeg * Math.PI) / 180;
  const lo = (lonDeg * Math.PI) / 180;
  return [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
}

function xyzToLatLon(v: [number, number, number]): [number, number] {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  const x = v[0] / n;
  const y = v[1] / n;
  const z = v[2] / n;
  return [(Math.asin(Math.max(-1, Math.min(1, z))) * 180) / Math.PI, (Math.atan2(y, x) * 180) / Math.PI];
}

const SIGMA = 8;
const K = 5;

interface RefSet {
  feats: number[][];
  xyz: [number, number, number][];
  priorXyz: [number, number, number];
}

function buildRefSet(p: Priors): RefSet {
  const raw = Buffer.from(p.featB64, 'base64');
  const dim = p.rows * p.cols * 3;
  const feats: number[][] = [];
  for (let i = 0; i < p.count; i += 1) {
    const f: number[] = [];
    for (let j = 0; j < dim; j += 1) {
      const q = raw[i * dim + j]!;
      const signed = q >= 128 ? q - 256 : q;
      f.push(signed / 32);
    }
    feats.push(f);
  }
  const xyz: [number, number, number][] = [];
  for (let i = 0; i < p.count; i += 1) {
    xyz.push(latLonToXyz(p.latE6[i]! / 1e6, p.lonE6[i]! / 1e6));
  }
  const priorXyz = latLonToXyz(p.priorLatE6 / 1e6, p.priorLonE6 / 1e6);
  return { feats, xyz, priorXyz };
}

/** Standardised 8x12 RGB grid, in the priors' feature space. */
function extractFeatures(jpeg: Uint8Array, p: Priors): number[] {
  const { frame, grids } = decodeDcGrid(jpeg);
  const rgb = rgbGrid(frame, grids);
  const g = downsample(rgb, p.rows, p.cols);
  const out: number[] = [];
  for (let j = 0; j < g.length; j += 1) {
    out.push((g[j]! - p.mean[j]!) / p.std[j]!);
  }
  return out;
}

function guess(feat: number[], ref: RefSet): [number, number] {
  // A photo with no reference to compare against falls back to the
  // geographic centre rather than crashing on an empty neighbour list.
  if (ref.feats.length === 0) return xyzToLatLon(ref.priorXyz);
  const dists: { d: number; i: number }[] = [];
  for (let i = 0; i < ref.feats.length; i += 1) {
    const f = ref.feats[i]!;
    let s = 0;
    for (let j = 0; j < feat.length; j += 1) {
      const dd = feat[j]! - f[j]!;
      s += dd * dd;
    }
    dists.push({ d: Math.sqrt(s), i });
  }
  dists.sort((a, b) => a.d - b.d);
  const top = dists.slice(0, Math.min(K, dists.length));
  let wsum = 0;
  let vx = 0;
  let vy = 0;
  let vz = 0;
  for (const t of top) {
    const w = Math.exp(-(t.d * t.d) / (2 * SIGMA * SIGMA));
    wsum += w;
    const v = ref.xyz[t.i]!;
    vx += w * v[0];
    vy += w * v[1];
    vz += w * v[2];
  }
  if (wsum > 0) {
    vx /= wsum;
    vy /= wsum;
    vz /= wsum;
  }
  const knnNorm = Math.hypot(vx, vy, vz) || 1;
  const conf = Math.exp(-(top[0]!.d * top[0]!.d) / (2 * SIGMA * SIGMA));
  const px = ref.priorXyz[0];
  const py = ref.priorXyz[1];
  const pz = ref.priorXyz[2];
  const fx = conf * (vx / knnNorm) + (1 - conf) * px;
  const fy = conf * (vy / knnNorm) + (1 - conf) * py;
  const fz = conf * (vz / knnNorm) + (1 - conf) * pz;
  return xyzToLatLon([fx, fy, fz]);
}

// ---------------------------------------------------------------------- env

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

async function main(): Promise<number> {
  const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
  // The brief is the one file of the instance the agent reads; confirm it is there.
  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  if (promptFile !== undefined && promptFile !== '') {
    try {
      readFileSync(promptFile, 'utf8');
    } catch {
      // The env still answers; continue without the brief.
    }
  }
  const priors = loadPriors();
  const ref = buildRefSet(priors);
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  const answers: { id: string; lat: number; lon: number }[] = [];
  for (const p of photos) {
    let lat = priors.priorLatE6 / 1e6;
    let lon = priors.priorLonE6 / 1e6;
    try {
      const res = await fetch(`${envUrl}${p.url}`);
      if (!res.ok) throw new Error(`GET ${p.url}: HTTP ${res.status}`);
      const jpeg = new Uint8Array(await res.arrayBuffer());
      const feat = extractFeatures(jpeg, priors);
      [lat, lon] = guess(feat, ref);
    } catch (err) {
      // A photo that cannot be read still gets the geographic centre.
      process.stderr.write(`geolocator: ${p.id} fell back to the prior: ${(err as Error).message}\n`);
    }
    if (!Number.isFinite(lat) || lat < -90 || lat > 90) lat = priors.priorLatE6 / 1e6;
    if (!Number.isFinite(lon) || lon < -180 || lon > 180) lon = priors.priorLonE6 / 1e6;
    answers.push({ id: p.id, lat: Math.round(lat * 1e6) / 1e6, lon: Math.round(lon * 1e6) / 1e6 });
  }
  if (answers.length === 0) return 0;
  const res = await fetch(`${envUrl}/answers`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ answers }),
  });
  process.stdout.write(`geolocator: ${answers.length} of ${photos.length} photos answered: HTTP ${res.status} ${await res.text()}`);
  return res.ok ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`geolocator: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
