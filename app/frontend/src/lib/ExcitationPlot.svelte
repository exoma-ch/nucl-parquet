<script>
  import Plotly from 'plotly.js-dist-min';

  let { z, a, projectile = 'p', targetA = null } = $props();

  let plotEl;
  let data = $state(null);
  let loading = $state(false);

  const LIB_COLORS = {
    'tendl-2024': '#6d9eeb', 'endfb-8.1': '#93c47d', 'jendl-5.0': '#ffa726',
    'jeff-3.3': '#ef5350', 'iaea-pd-2019': '#ab47bc',
  };

  async function load() {
    loading = true;
    let url = `/api/excitation/${z}/${a}?projectile=${projectile}`;
    if (targetA != null) url += `&target_a=${targetA}`;
    const res = await fetch(url);
    data = await res.json();
    loading = false;
    render();
  }

  function render() {
    if (!plotEl || !data) return;
    const traces = [];

    // Evaluated libraries
    for (const [lib, pts] of Object.entries(data.evaluated)) {
      traces.push({
        x: pts.map(p => p.E), y: pts.map(p => p.xs),
        mode: 'lines', name: lib,
        line: { color: LIB_COLORS[lib] || '#78909c', width: 2 },
      });
    }

    // EXFOR scatter with error bars
    if (data.exfor?.length) {
      const hasErr = data.exfor.some(p => p.err != null);
      traces.push({
        x: data.exfor.map(p => p.E), y: data.exfor.map(p => p.xs),
        error_y: hasErr ? {
          type: 'data', array: data.exfor.map(p => p.err ?? 0), visible: true,
          color: 'rgba(255,255,255,0.3)',
        } : undefined,
        mode: 'markers', name: 'EXFOR',
        marker: { size: 5, color: '#e0e0e0', opacity: 0.7, symbol: 'circle-open' },
        text: data.exfor.map(p => p.ref),
        hoverinfo: 'text+x+y',
      });
    }

    Plotly.react(plotEl, traces, {
      xaxis: { title: 'Energy (MeV)', gridcolor: '#2a2d38', zerolinecolor: '#2a2d38' },
      yaxis: { title: 'Cross section (mb)', gridcolor: '#2a2d38', zerolinecolor: '#2a2d38' },
      plot_bgcolor: '#1a1c25', paper_bgcolor: '#1a1c25',
      font: { color: '#c9cdd5', size: 11 },
      legend: { x: 0.7, y: 0.95, bgcolor: 'rgba(26,28,37,0.9)', bordercolor: '#2a2d38' },
      margin: { t: 10, r: 20, b: 40, l: 55 },
      height: 320,
    }, { responsive: true });
  }

  $effect(() => {
    // Reactively reload when z/a/projectile/targetA change
    void z; void a; void projectile; void targetA;
    if (z && a) load();
  });
</script>

<div class="excitation-plot">
  {#if loading}
    <p class="loading">Loading excitation function...</p>
  {:else if data && Object.keys(data.evaluated).length === 0 && (!data.exfor || data.exfor.length === 0)}
    <p style="color: var(--text-dim); font-size: 12px;">No cross-section data for this reaction</p>
  {/if}
  <div bind:this={plotEl}></div>
</div>

<style>
  .excitation-plot {
    background: var(--bg2); border-radius: 6px;
    border: 1px solid var(--border); padding: 8px; margin-top: 8px;
  }
</style>
