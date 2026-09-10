/* Numeric CSV handling shared by the viewer and regression tests. */
'use strict';
const PanelData = (() => {
  function number(value) {
    if (value == null || value.trim() === '') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    const header = lines.shift().replace(/^\uFEFF/, '').split(',').map(v => v.trim());
    if (!header.includes('frame') || !header.includes('distance_raw_m')) {
      throw new Error('Unsupported distance CSV: expected frame and distance_raw_m columns');
    }
    return lines.filter(line => line.trim()).map(line => {
      const values = line.split(',');
      const get = key => number(values[header.indexOf(key)]);
      return {frame: get('frame'), raw: get('distance_raw_m'),
              smoothed: get('distance_smoothed_m'), infer: get('inference_ms'), time: get('time_s')};
    }).filter(row => row.frame != null && row.frame >= 0);
  }
  function coverage(rows) {
    if (!rows.length) return null;
    const detected = rows.filter(row => row.raw != null).length;
    return {detected, total: rows.length, pct: Math.round(100 * detected / rows.length)};
  }
  function indexAtTime(rows, seconds, duration) {
    if (!rows.length) return -1;
    if (rows.every(row => row.time != null)) {
      let lo = 0, hi = rows.length;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (rows[mid].time <= seconds) lo = mid + 1; else hi = mid;
      }
      return Math.max(0, lo - 1);
    }
    const fps = Number.isFinite(duration) && duration > 0 ? rows.length / duration : 15;
    return Math.min(rows.length - 1, Math.max(0, Math.floor(seconds * fps)));
  }
  function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  }
  return {parseCSV, coverage, indexAtTime, escapeHTML};
})();
if (typeof module !== 'undefined') module.exports = PanelData;
