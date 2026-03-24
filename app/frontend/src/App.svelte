<script>
  import { onMount } from 'svelte';
  import Plotly from 'plotly.js-dist-min';
  import RangeInput from './lib/RangeInput.svelte';
  import ExcitationPlot from './lib/ExcitationPlot.svelte';
  import DecayChain from './lib/DecayChain.svelte';
  import CostEstimator from './lib/CostEstimator.svelte';
  import DoseKernel from './lib/DoseKernel.svelte';
  import YieldEstimator from './lib/YieldEstimator.svelte';

  // ── State ──
  let emissions = $state([]);
  let isotopes = $state([]);
  let selectedIso = $state(null);
  let detail = $state(null);
  let loading = $state(true);
  let detailLoading = $state(false);
  let sortCol = $state('electron_pct');
  let sortAsc = $state(false);

  // ── Filter state ──
  let hlMinLog = $state(0);      // log10(hours)
  let hlMaxLog = $state(2.9);
  // Per-type dose min/max (log10 MeV/Bq·s)
  let doseRange = $state({
    auger:   { lo: -4, hi: 0.7 },
    ce:      { lo: -4, hi: 0.7 },
    'beta-': { lo: -4, hi: 0.7 },
    alpha:   { lo: -4, hi: 0.7 },
    gamma:   { lo: -4, hi: 0.7 },
  });
  const DOSE_TYPES = [
    { key: 'auger', label: 'Auger', color: '#ffa726' },
    { key: 'ce', label: 'CE', color: '#93c47d' },
    { key: 'beta-', label: 'β⁻', color: '#6d9eeb' },
    { key: 'alpha', label: 'α', color: '#ef5350' },
    { key: 'gamma', label: 'γ', color: '#ab47bc' },
  ];
  let rangeMinLog = $state(-2);
  let rangeMaxLog = $state(6);

  let radTypes = $state({
    auger: true, ce: true, 'beta-': true, 'beta+/EC': false,
    alpha: true, gamma: true, xray: false,
  });
  let decayModes = $state({
    'B-': true, 'EC': true, 'B+': true, 'EC+B+': true,
    'A': true, 'IT': true, 'B-N': false, 'P': false,
  });

  // Excitation function state
  let selectedRoute = $state(null);

  const BANDS = [
    [0.001, 0.02, 'DNA', 'rgba(255,150,150,0.08)'],
    [0.02, 4, 'Nucleus', 'rgba(255,255,200,0.08)'],
    [4, 30, 'Cell', 'rgba(200,255,200,0.1)'],
    [30, 500, 'Cell cluster', 'rgba(200,240,255,0.1)'],
    [500, 1e4, 'Macrometastasis', 'rgba(180,220,255,0.1)'],
    [1e4, 2e5, 'Bulk tumor', 'rgba(230,200,255,0.08)'],
    [2e5, 2e6, 'Whole body', 'rgba(220,220,220,0.08)'],
  ];

  // ── Reference isotopes (clinical benchmarks) ──
  // Dose values: MeV/Bq·s (same unit as main table filters)
  const REF_ISOTOPES = [
    // Therapeutic β⁻
    { iso: 'Lu-177',  hl: '6.6d',  decay: 'β⁻', type: 'Therapeutic', ce: 0.0113, auger: 0.0006, beta: 0.0472, alpha: 0, gamma: 0.0108, use: 'PSMA, DOTATATE (Lutathera/Pluvicto)' },
    { iso: 'Y-90',    hl: '2.7d',  decay: 'β⁻', type: 'Therapeutic', ce: 0, auger: 0, beta: 0.3267, alpha: 0, gamma: 0, use: 'Microspheres (liver), anti-CD20' },
    { iso: 'I-131',   hl: '8.0d',  decay: 'β⁻', type: 'Theranostic', ce: 0.0047, auger: 0.0003, beta: 0.0641, alpha: 0, gamma: 0.0568, use: 'Thyroid cancer/ablation' },
    { iso: 'Re-188',  hl: '17.0h', decay: 'β⁻', type: 'Therapeutic', ce: 0.0006, auger: 0.0001, beta: 0.2116, alpha: 0, gamma: 0.0091, use: 'Generator-produced, bone pain' },
    { iso: 'Sm-153',  hl: '1.9d',  decay: 'β⁻', type: 'Therapeutic', ce: 0.0011, auger: 0.0002, beta: 0.0779, alpha: 0, gamma: 0.0051, use: 'Bone pain palliation (Quadramet)' },
    { iso: 'Ho-166',  hl: '1.1d',  decay: 'β⁻', type: 'Therapeutic', ce: 0.0002, auger: 0, beta: 0.1858, alpha: 0, gamma: 0.0025, use: 'Liver, microspheres' },
    // Therapeutic α
    { iso: 'Ac-225',  hl: '10.0d', decay: 'α', type: 'Therapeutic', ce: 0, auger: 0.0008, beta: 0, alpha: 0.4051, gamma: 0.0014, use: 'PSMA-617, emerging α therapy' },
    { iso: 'Ra-223',  hl: '11.4d', decay: 'α', type: 'Therapeutic', ce: 0, auger: 0.0003, beta: 0.0012, alpha: 0.3862, gamma: 0.0019, use: 'Bone metastases (Xofigo)' },
    { iso: 'At-211',  hl: '7.2h',  decay: 'α/EC', type: 'Therapeutic', ce: 0, auger: 0.0015, beta: 0, alpha: 0.4325, gamma: 0.0038, use: 'Brain tumors, single α' },
    { iso: 'Bi-213',  hl: '45.6m', decay: 'α/β⁻', type: 'Therapeutic', ce: 0.0001, auger: 0.0002, beta: 0.0312, alpha: 0.3605, gamma: 0.0182, use: 'Leukemia, short-range α' },
    // Auger / CE
    { iso: 'In-111',  hl: '2.8d',  decay: 'EC', type: 'Theranostic', ce: 0.0097, auger: 0.0032, beta: 0, alpha: 0, gamma: 0.0521, use: 'Octreotide SPECT, Auger therapy' },
    { iso: 'I-125',   hl: '59.4d', decay: 'EC', type: 'Therapeutic', ce: 0.0037, auger: 0.0019, beta: 0, alpha: 0, gamma: 0.0054, use: 'Brachytherapy seeds, Auger' },
    { iso: 'Sn-117m', hl: '14.0d', decay: 'IT', type: 'Therapeutic', ce: 0.0351, auger: 0.0009, beta: 0, alpha: 0, gamma: 0.0201, use: 'CE therapy, bone pain' },
    // Diagnostic PET
    { iso: 'F-18',    hl: '1.8h',  decay: 'β⁺', type: 'Diagnostic', ce: 0, auger: 0, beta: 0.0851, alpha: 0, gamma: 0.1396, use: 'FDG-PET (gold standard)' },
    { iso: 'Ga-68',   hl: '1.1h',  decay: 'β⁺', type: 'Diagnostic', ce: 0.0001, auger: 0.0003, beta: 0.2927, alpha: 0, gamma: 0.1270, use: 'DOTATATE/PSMA PET' },
    { iso: 'Cu-64',   hl: '12.7h', decay: 'β⁺/β⁻', type: 'Theranostic', ce: 0, auger: 0.0002, beta: 0.0388, alpha: 0, gamma: 0.0247, use: 'Emerging theranostic PET' },
    { iso: 'Zr-89',   hl: '3.3d',  decay: 'β⁺/EC', type: 'Diagnostic', ce: 0, auger: 0.0004, beta: 0.0117, alpha: 0, gamma: 0.1613, use: 'Immuno-PET (long t½)' },
    // Diagnostic SPECT
    { iso: 'Tc-99m',  hl: '6.0h',  decay: 'IT', type: 'Diagnostic', ce: 0.0113, auger: 0.0008, beta: 0, alpha: 0, gamma: 0.0163, use: 'Workhorse SPECT (generator)' },
  ];
  let showRefTable = $state(false);

  const MARKER_MAP = {
    auger: { symbol: 'triangle-down', name: 'Auger e⁻', color: '#ffa726' },
    ce: { symbol: 'square', name: 'Conv. e⁻', color: '#93c47d' },
    'beta-': { symbol: 'circle', name: 'β⁻', color: '#6d9eeb' },
    'beta+/EC': { symbol: 'diamond', name: 'β⁺/EC', color: '#42a5f5' },
    alpha: { symbol: 'triangle-up', name: 'α', color: '#ef5350' },
    gamma: { symbol: 'star', name: 'γ', color: '#ab47bc' },
    xray: { symbol: 'x', name: 'X-ray', color: '#78909c' },
  };

  let suppliers = $state([]);

  // ── Format / parse for RangeInput ──
  // Half-life: display in hours, internal = log10(hours)
  function hlFormat(v) {
    const h = 10 ** v;
    if (h < 1) return (h * 60).toFixed(0);
    if (h < 24) return h.toFixed(1);
    return (h / 24).toFixed(1);
  }
  function hlUnit(v) {
    const h = 10 ** v;
    if (h < 1) return 'min';
    if (h < 24) return 'h';
    return 'd';
  }
  // Parse: accepts "30m", "6h", "2.5d", or bare number (uses current display unit)
  function hlParse(text) {
    const m = text.trim().match(/^([0-9.e+-]+)\s*(s|m|min|h|d)?$/i);
    if (!m) return null;
    const n = parseFloat(m[1]);
    if (isNaN(n) || n <= 0) return null;
    const u = (m[2] || '').toLowerCase();
    if (u === 's') return Math.log10(n / 3600);
    if (u === 'm' || u === 'min') return Math.log10(n / 60);
    if (u === 'd') return Math.log10(n * 24);
    return Math.log10(n); // bare number = hours
  }

  // Dose: display as scientific notation, internal = log10(MeV/Bq·s)
  function doseFormat(v) {
    return (10 ** v).toExponential(1);
  }
  function doseParse(text) {
    const n = parseFloat(text.trim());
    if (isNaN(n) || n <= 0) return null;
    return Math.log10(n);
  }

  // Range: display in μm, internal = log10(μm)
  function rangeFormat(v) {
    const um = 10 ** v;
    if (um < 0.1) return (um * 1000).toFixed(0);
    if (um < 1000) return um.toFixed(1);
    if (um < 1e4) return (um / 1000).toFixed(1);
    if (um < 1e6) return (um / 1e4).toFixed(1);
    return (um / 1e6).toFixed(1);
  }
  function rangeUnit(v) {
    const um = 10 ** v;
    if (um < 0.1) return 'nm';
    if (um < 1000) return 'μm';
    if (um < 1e4) return 'mm';
    if (um < 1e6) return 'cm';
    return 'm';
  }
  // Parse: accepts "200nm", "5μm", "1.2mm", "3cm", "0.5m", or bare number (= μm)
  function rangeParse(text) {
    const m = text.trim().match(/^([0-9.e+-]+)\s*(nm|μm|um|mm|cm|m)?$/i);
    if (!m) return null;
    const n = parseFloat(m[1]);
    if (isNaN(n) || n <= 0) return null;
    const u = (m[2] || '').toLowerCase();
    if (u === 'nm') return Math.log10(n / 1000);
    if (u === 'mm') return Math.log10(n * 1000);
    if (u === 'cm') return Math.log10(n * 1e4);
    if (u === 'm') return Math.log10(n * 1e6);
    return Math.log10(n); // bare number or μm/um
  }

  function activeRad() {
    return Object.entries(radTypes).filter(([, v]) => v).map(([k]) => k).join(',');
  }
  function activeDecay() {
    return Object.entries(decayModes).filter(([, v]) => v).map(([k]) => k).join(',');
  }

  // ── Fetch ──
  function globalDoseMin() {
    return Math.min(...Object.values(doseRange).map(r => r.lo));
  }

  function doseInRange(type, dose) {
    const r = doseRange[type];
    if (!r) return true;
    const lo = 10 ** r.lo, hi = 10 ** r.hi;
    return dose >= lo && dose <= hi;
  }

  // Map isotope table keys to emission rad_type keys
  const ISO_TYPE_MAP = { ce: 'ce', auger: 'auger', beta: 'beta-', alpha: 'alpha', gamma: 'gamma' };

  async function fetchData() {
    loading = true;
    const params = new URLSearchParams({
      hl_min_h: (10 ** hlMinLog).toFixed(4),
      hl_max_h: (10 ** hlMaxLog).toFixed(4),
      dose_min: (10 ** globalDoseMin()).toFixed(6),
      range_min: (10 ** rangeMinLog).toFixed(4),
      range_max: (10 ** rangeMaxLog).toFixed(4),
      rad_types: activeRad(),
      decay_modes: activeDecay(),
    });
    const [emRes, isoRes] = await Promise.all([
      fetch(`/api/emissions?${params}`),
      fetch(`/api/isotopes?hl_min_h=${(10**hlMinLog).toFixed(4)}&hl_max_h=${(10**hlMaxLog).toFixed(4)}&decay_modes=${activeDecay()}`),
    ]);

    // Per-type dose range filtering on emissions
    const allEmissions = await emRes.json();
    emissions = allEmissions.filter(e => doseInRange(e.rad_type, e.dose));

    // Per-type dose range filtering on isotopes (keep if any type in range)
    const allIsotopes = await isoRes.json();
    isotopes = allIsotopes.filter(iso =>
      Object.entries(ISO_TYPE_MAP).some(([isoKey, rtKey]) => {
        const d = iso[isoKey];
        return d > 0 && doseInRange(rtKey, d);
      })
    );

    loading = false;
    renderPlot();
  }

  async function fetchDetail(z, a) {
    detailLoading = true;
    selectedRoute = null;
    const res = await fetch(`/api/isotope/${z}/${a}`);
    detail = await res.json();
    detailLoading = false;
  }

  // ── Plot ──
  let plotEl;

  function renderPlot() {
    if (!plotEl || !emissions.length) return;

    const traces = [];
    const shapes = [];
    const annotations = [];

    // Bio bands
    for (const [x0, x1, label, color] of BANDS) {
      shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0, x1, y0: 0, y1: 1, fillcolor: color, line: { width: 0 } });
      annotations.push({
        x: Math.log10(Math.sqrt(x0 * x1)), y: 1.08, xref: 'x', yref: 'paper',
        text: label, showarrow: false, font: { size: 10, color: '#666' }, textangle: -90,
      });
    }

    // Group by rad_type
    const byType = {};
    for (const e of emissions) {
      (byType[e.rad_type] ??= []).push(e);
    }

    for (const [rt, pts] of Object.entries(byType)) {
      const m = MARKER_MAP[rt];
      if (!m) continue;
      traces.push({
        x: pts.map(p => p.range_um),
        y: pts.map(p => p.dose),
        mode: 'markers',
        type: 'scatter',
        marker: { symbol: m.symbol, size: 7, color: m.color, opacity: 0.6, line: { width: 0.3, color: '#000' } },
        text: pts.map(p => `${p.isotope} (${p.hl_label})<br>${p.rad_subtype || rt}: ${p.energy_keV} keV<br>I=${p.intensity_pct}%<br>Dose: ${p.dose} MeV/Bq·s<br>Range: ${p.range_um.toFixed(1)} μm`),
        hoverinfo: 'text',
        name: m.name,
        customdata: pts.map(p => [p.Z, p.A, p.isotope]),
      });
    }

    const layout = {
      xaxis: {
        type: 'log', title: { text: 'CSDA range / MFP in water (μm)', font: { size: 13 } },
        range: [rangeMinLog, rangeMaxLog],
        tickvals: [0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6],
        ticktext: ['10nm', '100nm', '1μm', '10μm', '100μm', '1mm', '1cm', '10cm', '1m'],
        gridcolor: '#2a2d38', zerolinecolor: '#2a2d38',
      },
      yaxis: {
        type: 'log', title: { text: 'Dose rate (MeV / Bq·s)', font: { size: 13 } },
        range: [globalDoseMin(), 0.7],
        gridcolor: '#2a2d38', zerolinecolor: '#2a2d38',
      },
      shapes, annotations,
      plot_bgcolor: '#1a1c25', paper_bgcolor: '#1a1c25',
      font: { color: '#c9cdd5', size: 11 },
      legend: { x: 0.01, y: 0.01, bgcolor: 'rgba(26,28,37,0.9)', bordercolor: '#2a2d38', borderwidth: 1 },
      margin: { t: 30, r: 20, b: 50, l: 60 },
      height: 550,
    };

    Plotly.react(plotEl, traces, layout, { responsive: true });

    plotEl.on('plotly_click', (data) => {
      if (data.points[0]?.customdata) {
        const [z, a, iso] = data.points[0].customdata;
        selectedIso = { Z: z, A: a, isotope: iso };
        fetchDetail(z, a);
      }
    });
  }

  // ── Sort ──
  function sortBy(col) {
    if (sortCol === col) sortAsc = !sortAsc;
    else { sortCol = col; sortAsc = false; }
  }

  function sortedIsotopes() {
    return [...isotopes].sort((a, b) => {
      const va = a[sortCol] ?? 0, vb = b[sortCol] ?? 0;
      return sortAsc ? va - vb : vb - va;
    });
  }

  // ── Lifecycle ──
  onMount(() => {
    fetchData();
    fetch('/api/suppliers').then(r => r.json()).then(d => suppliers = d);
  });

  // Debounced re-fetch
  let fetchTimer;
  function debouncedFetch() {
    clearTimeout(fetchTimer);
    fetchTimer = setTimeout(fetchData, 300);
  }
</script>

<div id="app">
  <h1>Therapeutic Isotope Explorer</h1>
  <p class="subtitle">
    Dose deposition landscape — ranges from NIST ESTAR (e⁻), ASTAR (α), XCOM (γ) in water
    &middot; {emissions.length} emissions from {isotopes.length} isotopes
  </p>

  <!-- ── Reference isotope table ── -->
  <button class="ref-toggle" onclick={() => showRefTable = !showRefTable}>
    {showRefTable ? '▾' : '▸'} Clinical Reference Isotopes
  </button>
  {#if showRefTable}
    <div class="table-container ref-table" style="max-height: 420px; overflow-y: auto;">
      <table>
        <thead>
          <tr>
            <th>Isotope</th><th>t½</th><th>Decay</th><th>Role</th>
            <th>CE</th><th>Auger</th><th>β⁻</th><th>α</th><th>γ</th>
            <th>Total</th><th>Clinical use</th>
          </tr>
        </thead>
        <tbody>
          {#each REF_ISOTOPES as r}
            {@const total = r.ce + r.auger + r.beta + r.alpha + r.gamma}
            <tr>
              <td style="font-weight: 700; color: #e8eaed;">{r.iso}</td>
              <td>{r.hl}</td>
              <td>{r.decay}</td>
              <td>
                <span class="badge {r.type === 'Diagnostic' ? 'badge-diag' : r.type === 'Theranostic' ? 'badge-theranostic' : 'badge-ther'}">{r.type}</span>
              </td>
              <td style="color: #93c47d;">{r.ce > 0 ? r.ce.toFixed(4) : '—'}</td>
              <td style="color: #ffa726;">{r.auger > 0 ? r.auger.toFixed(4) : '—'}</td>
              <td style="color: #6d9eeb;">{r.beta > 0 ? r.beta.toFixed(4) : '—'}</td>
              <td style="color: #ef5350;">{r.alpha > 0 ? r.alpha.toFixed(4) : '—'}</td>
              <td style="color: #ab47bc;">{r.gamma > 0 ? r.gamma.toFixed(4) : '—'}</td>
              <td style="font-weight: 600;">{total.toFixed(4)}</td>
              <td style="font-size: 10px; max-width: 200px;">{r.use}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  <!-- ── Filters ── -->
  <div class="filters">
    <RangeInput
      label="Half-life"
      min={-1} max={4} step={0.05}
      bind:low={hlMinLog} bind:high={hlMaxLog}
      formatValue={hlFormat}
      parseValue={hlParse}
      unit={hlUnit}
      oninput={debouncedFetch}
    />
    {#each DOSE_TYPES as dt}
      <RangeInput
        label="{dt.label} dose"
        min={-4} max={0.7} step={0.1}
        bind:low={doseRange[dt.key].lo} bind:high={doseRange[dt.key].hi}
        formatValue={doseFormat}
        parseValue={doseParse}
        unit="MeV/Bq·s"
        oninput={debouncedFetch}
      />
    {/each}
    <RangeInput
      label="Range"
      min={-2} max={6} step={0.1}
      bind:low={rangeMinLog} bind:high={rangeMaxLog}
      formatValue={rangeFormat}
      parseValue={rangeParse}
      unit={rangeUnit}
      oninput={debouncedFetch}
    />
    <div class="filter-group">
      <label>Emission types</label>
      <div class="checkbox-group">
        {#each Object.keys(radTypes) as rt}
          <label>
            <input type="checkbox" bind:checked={radTypes[rt]} onchange={debouncedFetch} />
            {MARKER_MAP[rt]?.name ?? rt}
          </label>
        {/each}
      </div>
    </div>
    <div class="filter-group">
      <label>Decay modes</label>
      <div class="checkbox-group">
        {#each Object.keys(decayModes) as dm}
          <label>
            <input type="checkbox" bind:checked={decayModes[dm]} onchange={debouncedFetch} />
            {dm}
          </label>
        {/each}
      </div>
    </div>
  </div>

  <!-- ── Plot ── -->
  <div class="plot-container">
    <div bind:this={plotEl}></div>
    {#if loading}<p class="loading">Loading emission data...</p>{/if}
  </div>

  <!-- ── Isotope Table ── -->
  <h2>Isotope Summary (sorted by {sortCol})</h2>
  <div class="table-container" style="max-height: 400px; overflow-y: auto;">
    <table>
      <thead>
        <tr>
          <th onclick={() => sortBy('isotope')}>Isotope</th>
          <th onclick={() => sortBy('half_life_s')}>Half-life</th>
          <th onclick={() => sortBy('primary_decay')}>Decay</th>
          <th onclick={() => sortBy('total')}>Total dose</th>
          <th onclick={() => sortBy('ce')}>CE</th>
          <th onclick={() => sortBy('auger')}>Auger</th>
          <th onclick={() => sortBy('beta')}>β⁻</th>
          <th onclick={() => sortBy('alpha')}>α</th>
          <th onclick={() => sortBy('gamma')}>γ</th>
          <th onclick={() => sortBy('electron_pct')}>e⁻ %</th>
        </tr>
      </thead>
      <tbody>
        {#each sortedIsotopes().slice(0, 100) as iso}
          <tr
            class:selected={selectedIso?.Z === iso.Z && selectedIso?.A === iso.A}
            onclick={() => { selectedIso = iso; fetchDetail(iso.Z, iso.A); }}
          >
            <td style="font-weight: 600; color: #e8eaed;">{iso.isotope}</td>
            <td>{iso.hl_label}</td>
            <td>{iso.primary_decay}</td>
            <td>{iso.total.toFixed(4)}</td>
            <td>{iso.ce > 0 ? iso.ce.toFixed(4) : '—'}</td>
            <td>{iso.auger > 0 ? iso.auger.toFixed(4) : '—'}</td>
            <td>{iso.beta > 0 ? iso.beta.toFixed(4) : '—'}</td>
            <td>{iso.alpha > 0 ? iso.alpha.toFixed(4) : '—'}</td>
            <td>{iso.gamma > 0 ? iso.gamma.toFixed(4) : '—'}</td>
            <td style="color: var(--accent); font-weight: 600;">{iso.electron_pct}%</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <!-- ── Detail Panel ── -->
  {#if detailLoading}
    <div class="detail-panel"><p class="loading">Loading detail...</p></div>
  {:else if detail}
    <div class="detail-panel">
      <button class="close-btn" onclick={() => { detail = null; selectedIso = null; selectedRoute = null; }}>Close</button>
      <h2>{detail.isotope}</h2>
      {#if detail.ground_state}
        <p>
          <strong>Half-life:</strong> {detail.ground_state.hl_label}
          &middot; <strong>J<sup>π</sup>:</strong> {detail.ground_state.jp ?? '?'}
          &middot; <strong>Decay:</strong> {detail.ground_state.decay_1} ({detail.ground_state.decay_1_pct}%)
          {#if detail.ground_state.decay_2}
            , {detail.ground_state.decay_2} ({detail.ground_state.decay_2_pct}%)
          {/if}
        </p>
      {/if}

      <div class="detail-grid">
        <!-- Targets -->
        <div class="detail-section">
          <h3>Natural Target Abundances</h3>
          <table>
            <thead><tr><th>Target</th><th>Abundance</th><th>Mass (u)</th></tr></thead>
            <tbody>
              {#each detail.targets as t}
                <tr>
                  <td>{t.symbol}-{t.A} (Z={t.Z})</td>
                  <td>{t.abundance}%</td>
                  <td>{t.mass}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>

        <!-- Production routes (clickable for excitation) -->
        <div class="detail-section">
          <h3>Production Routes (evaluated) <span class="click-hint">click for σ(E)</span></h3>
          {#if detail.routes.length}
            <table>
              <thead><tr><th>Library</th><th>Reaction</th><th>Peak σ</th><th>Peak E</th></tr></thead>
              <tbody>
                {#each detail.routes.slice(0, 15) as r}
                  <tr
                    class="route-row"
                    class:route-selected={selectedRoute === r}
                    onclick={() => { selectedRoute = selectedRoute === r ? null : r; }}
                  >
                    <td>{r.library}</td>
                    <td>{r.projectile} + {r.target_el}-{r.target_A}</td>
                    <td>{r.peak_xs} mb</td>
                    <td>{r.peak_E} MeV</td>
                  </tr>
                {/each}
              </tbody>
            </table>
            {#if selectedRoute}
              <ExcitationPlot
                z={detail.Z} a={detail.A}
                projectile={selectedRoute.projectile}
                targetA={selectedRoute.target_A}
              />
            {/if}
          {:else}
            <p style="color: var(--text-dim);">No evaluated routes found</p>
          {/if}
        </div>

        <!-- EXFOR -->
        <div class="detail-section">
          <h3>EXFOR Experimental</h3>
          {#if detail.exfor.length}
            <table>
              <thead><tr><th>Reaction</th><th>Entries</th><th>Peak σ</th><th>References</th></tr></thead>
              <tbody>
                {#each detail.exfor.slice(0, 10) as e}
                  <tr>
                    <td>{e.projectile}+{e.target_el}-{e.target_A}</td>
                    <td>{e.n_entries}</td>
                    <td>{e.peak_xs} mb</td>
                    <td style="font-size: 10px; max-width: 200px; overflow: hidden; text-overflow: ellipsis;">{e.refs}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {:else}
            <p style="color: var(--text-dim);">No EXFOR data found</p>
          {/if}
        </div>

        <!-- Emissions -->
        <div class="detail-section">
          <h3>Decay Emissions</h3>
          <table>
            <thead><tr><th>Type</th><th>Subtype</th><th>E (keV)</th><th>I (%)</th><th>Dose</th></tr></thead>
            <tbody>
              {#each detail.emissions.slice(0, 20) as em}
                <tr>
                  <td><span class="badge badge-{em.rad_type === 'ce' ? 'ce' : em.rad_type === 'auger' ? 'auger' : em.rad_type === 'alpha' ? 'alpha' : em.rad_type === 'gamma' ? 'gamma' : 'beta'}">{em.rad_type}</span></td>
                  <td>{em.subtype || '—'}</td>
                  <td>{em.energy_keV}</td>
                  <td>{em.intensity_pct}</td>
                  <td>{em.dose}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>

        <!-- Decay Chain -->
        <div class="detail-section">
          <h3>Decay Chain & Dose Contributions</h3>
          <DecayChain z={detail.Z} a={detail.A} />
        </div>

        <!-- Dose Kernel -->
        <div class="detail-section detail-wide">
          <h3>Dose Kernel — D(r) Radial Distribution</h3>
          <DoseKernel z={detail.Z} a={detail.A} />
        </div>

        <!-- Cost Estimator -->
        <div class="detail-section">
          <h3>Production Feasibility</h3>
          <CostEstimator z={detail.Z} a={detail.A} />
        </div>

        <!-- Yield Estimator -->
        <div class="detail-section">
          <h3>Yield Estimator</h3>
          <YieldEstimator z={detail.Z} a={detail.A} route={selectedRoute} />
        </div>

        <!-- Theranostic partners -->
        <div class="detail-section">
          <h3>Theranostic Partners (Z={detail.Z})</h3>
          {#if detail.partners.length}
            <table>
              <thead><tr><th>Isotope</th><th>Half-life</th><th>Decay</th><th>Role</th></tr></thead>
              <tbody>
                {#each detail.partners as p}
                  <tr>
                    <td style="font-weight: 600;">{p.isotope}</td>
                    <td>{p.hl_label}</td>
                    <td>{p.decay}</td>
                    <td>
                      {#if p.role.includes('diagnostic')}<span class="badge badge-diag">{p.role}</span>
                      {:else if p.role.includes('therapy')}<span class="badge badge-ther">{p.role}</span>
                      {:else}{p.role}
                      {/if}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {:else}
            <p style="color: var(--text-dim);">No same-element partners found</p>
          {/if}
        </div>

        <!-- Suppliers -->
        <div class="detail-section">
          <h3>Enriched Target Suppliers</h3>
          {#if suppliers.length}
            <table>
              <thead><tr><th>Supplier</th><th>Type</th><th>Notes</th><th>Status</th></tr></thead>
              <tbody>
                {#each suppliers as s}
                  <tr>
                    <td><a class="supplier-link" href={s.url} target="_blank" rel="noreferrer">{s.name}</a></td>
                    <td style="font-size: 10px;">{s.type?.replace('_', ' ') ?? ''}</td>
                    <td>{s.note}</td>
                    <td>
                      {#if s.status === 'ok'}<span style="color: #93c47d;">●</span>
                      {:else if s.status === 'down'}<span style="color: #ef5350;">●</span>
                      {:else}<span style="color: var(--text-dim);">?</span>
                      {/if}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {:else}
            <p class="loading">Loading suppliers...</p>
          {/if}
          {#if detail.targets.length}
            {@const top = detail.targets[0]}
            <p style="margin-top: 8px; font-size: 12px;">
              Primary target: <strong>{top.symbol}-{top.A}</strong> ({top.abundance}% nat. abundance)
              {#if top.abundance < 10}
                — <span style="color: #ffa726;">enrichment required</span>
              {:else if top.abundance > 90}
                — <span style="color: #93c47d;">natural target viable</span>
              {/if}
            </p>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>
