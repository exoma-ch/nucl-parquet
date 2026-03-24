<script>
  import Plotly from 'plotly.js-dist-min';

  let { z, a, route = null } = $props();

  let mode = $state('cyclotron');
  let loading = $state(false);
  let result = $state(null);
  let plotEl = $state();

  // Cyclotron params
  let beamCurrent_uA = $state(10);
  let eIn_MeV = $state(30);
  let eOut_MeV = $state(5);
  let enrichment = $state(0.99);
  let irradTime_h = $state(4);
  let targetMass_mg = $state(100);

  // Reactor params
  let flux = $state(1e14);
  let targetMass_r_mg = $state(100);
  let enrichment_r = $state(0.99);
  let irradTime_r_h = $state(24);

  // Pre-fill from route
  $effect(() => {
    if (route) {
      if (route.peak_E) eIn_MeV = Math.min(route.peak_E * 1.3, route.E_max || 100);
      if (route.E_min) eOut_MeV = route.E_min;
      if (route.projectile === 'n') mode = 'reactor';
      else mode = 'cyclotron';
    }
  });

  async function calculate() {
    loading = true;
    const body = mode === 'cyclotron' ? {
      mode: 'cyclotron',
      z, a,
      projectile: route?.projectile || 'p',
      target_Z: route?.target_Z || z,
      target_A: route?.target_A || a - 1,
      beam_current_uA: beamCurrent_uA,
      energy_in_MeV: eIn_MeV,
      energy_out_MeV: eOut_MeV,
      enrichment,
      irrad_time_h: irradTime_h,
      target_mass_mg: targetMass_mg,
    } : {
      mode: 'reactor',
      z, a,
      target_Z: route?.target_Z || z,
      target_A: route?.target_A || a - 1,
      flux_n_cm2_s: flux,
      target_mass_mg: targetMass_r_mg,
      enrichment: enrichment_r,
      irrad_time_h: irradTime_r_h,
    };

    const res = await fetch('/api/yield-estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    result = await res.json();
    loading = false;
    if (result.integrand && plotEl) renderIntegrand();
  }

  function renderIntegrand() {
    if (!result.integrand || !plotEl) return;
    const { E, sigma, S, integrand } = result.integrand;
    Plotly.react(plotEl, [
      { x: E, y: sigma, mode: 'lines', name: 'σ (mb)', yaxis: 'y', line: { color: '#6d9eeb', width: 2 } },
      { x: E, y: S, mode: 'lines', name: 'S (MeV cm²/g)', yaxis: 'y2', line: { color: '#ffa726', width: 1.5, dash: 'dot' } },
      { x: E, y: integrand, mode: 'lines', name: 'σ/S integrand', yaxis: 'y3',
        line: { color: '#93c47d', width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(147,196,125,0.15)' },
    ], {
      xaxis: { title: 'Energy (MeV)', gridcolor: '#2a2d38', zerolinecolor: '#2a2d38' },
      yaxis: { title: 'σ (mb)', titlefont: { color: '#6d9eeb' }, gridcolor: '#2a2d38', side: 'left' },
      yaxis2: { title: 'S', titlefont: { color: '#ffa726' }, overlaying: 'y', side: 'right', showgrid: false },
      yaxis3: { title: 'σ/S', titlefont: { color: '#93c47d' }, overlaying: 'y', side: 'right', position: 0.95, showgrid: false, visible: false },
      plot_bgcolor: '#1a1c25', paper_bgcolor: '#1a1c25',
      font: { color: '#c9cdd5', size: 11 },
      legend: { x: 0.01, y: 0.95, bgcolor: 'rgba(26,28,37,0.9)', bordercolor: '#2a2d38' },
      margin: { t: 10, r: 60, b: 40, l: 55 },
      height: 280,
    }, { responsive: true });
  }

  function fmtActivity(bq) {
    if (bq == null || isNaN(bq)) return '—';
    if (bq >= 1e9) return `${(bq / 1e9).toFixed(2)} GBq`;
    if (bq >= 1e6) return `${(bq / 1e6).toFixed(2)} MBq`;
    if (bq >= 1e3) return `${(bq / 1e3).toFixed(2)} kBq`;
    return `${bq.toFixed(1)} Bq`;
  }
</script>

<div class="yield-estimator">
  <div class="yield-tabs">
    <button class:active={mode === 'cyclotron'} onclick={() => mode = 'cyclotron'}>Cyclotron</button>
    <button class:active={mode === 'reactor'} onclick={() => mode = 'reactor'}>Reactor</button>
  </div>

  {#if mode === 'cyclotron'}
    <div class="yield-form">
      <label>Beam current (μA)
        <input type="number" bind:value={beamCurrent_uA} min="0.1" step="1" />
      </label>
      <label>E<sub>in</sub> (MeV)
        <input type="number" bind:value={eIn_MeV} min="1" step="1" />
      </label>
      <label>E<sub>out</sub> (MeV)
        <input type="number" bind:value={eOut_MeV} min="0" step="1" />
      </label>
      <label>Enrichment
        <input type="number" bind:value={enrichment} min="0.01" max="1" step="0.01" />
      </label>
      <label>Irrad. time (h)
        <input type="number" bind:value={irradTime_h} min="0.1" step="0.5" />
      </label>
      <label>Target mass (mg)
        <input type="number" bind:value={targetMass_mg} min="1" step="10" />
      </label>
    </div>
  {:else}
    <div class="yield-form">
      <label>Flux (n/cm²·s)
        <input type="number" bind:value={flux} min="1e10" step="1e13" />
      </label>
      <label>Target mass (mg)
        <input type="number" bind:value={targetMass_r_mg} min="1" step="10" />
      </label>
      <label>Enrichment
        <input type="number" bind:value={enrichment_r} min="0.01" max="1" step="0.01" />
      </label>
      <label>Irrad. time (h)
        <input type="number" bind:value={irradTime_r_h} min="0.1" step="1" />
      </label>
    </div>
  {/if}

  <button class="yield-calc-btn" onclick={calculate} disabled={loading}>
    {loading ? 'Calculating...' : 'Calculate Yield'}
  </button>

  {#if result}
    <div class="yield-result">
      <div class="yield-stat">
        <span class="yield-value">{fmtActivity(result.activity_Bq)}</span>
        <span class="yield-label">EOB Activity</span>
      </div>
      {#if result.yield_atoms != null}
        <div class="yield-stat">
          <span class="yield-value">{result.yield_atoms.toExponential(2)}</span>
          <span class="yield-label">Atoms produced</span>
        </div>
      {/if}
      {#if result.specific_activity_GBq_umol != null}
        <div class="yield-stat">
          <span class="yield-value">{result.specific_activity_GBq_umol.toFixed(2)}</span>
          <span class="yield-label">GBq/μmol</span>
        </div>
      {/if}
    </div>
    {#if result.integrand}
      <div bind:this={plotEl} style="margin-top: 8px;"></div>
    {/if}
    {#if result.warning}
      <p class="yield-warning">{result.warning}</p>
    {/if}
  {/if}
</div>

<style>
  .yield-estimator { margin-top: 8px; }
  .yield-tabs { display: flex; gap: 4px; margin-bottom: 10px; }
  .yield-tabs button {
    padding: 4px 14px; font-size: 12px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text-dim); border-radius: 4px;
    cursor: pointer; transition: all 0.15s;
  }
  .yield-tabs button.active {
    background: var(--accent); color: #fff; border-color: var(--accent);
  }
  .yield-form {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 8px; margin-bottom: 10px;
  }
  .yield-form label {
    display: flex; flex-direction: column; gap: 2px;
    font-size: 11px; color: var(--text-dim);
  }
  .yield-form input {
    width: 100%; font-size: 12px; font-family: var(--mono);
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); padding: 4px 6px;
  }
  .yield-form input:focus { outline: 1px solid var(--accent); }
  .yield-calc-btn {
    padding: 6px 18px; font-size: 12px; font-weight: 600;
    background: var(--accent); color: #fff; border: none; border-radius: 4px;
    cursor: pointer;
  }
  .yield-calc-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .yield-result {
    display: flex; gap: 20px; margin-top: 12px; flex-wrap: wrap;
  }
  .yield-stat { display: flex; flex-direction: column; }
  .yield-value {
    font-size: 18px; font-weight: 700; color: #e8eaed; font-family: var(--mono);
  }
  .yield-label { font-size: 11px; color: var(--text-dim); }
  .yield-warning {
    margin-top: 8px; font-size: 11px; color: #ffa726; font-style: italic;
  }
</style>
